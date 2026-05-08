import os
import time
from typing import Dict, Any, List
from .models import StoryboardFrame, Character, Scene, Prop, GenerationStatus, ImageAsset, ImageVariant
from ...models.image import WanxImageModel
from ...utils import get_logger
from ...utils.oss_utils import is_object_key

logger = get_logger(__name__)

class StoryboardGenerator:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = self._build_model(self.config.get('model', {}))
        self.output_dir = self.config.get('output_dir', 'output/storyboard')

    @staticmethod
    def _build_model(model_config: Dict[str, Any]):
        """Pick image-gen backend based on IMAGE_PROVIDER env var.

        Mirrors AssetGenerator._build_model so storyboard frame renders go
        through the same local runtime (and the same singleton manager,
        sharing pipe state + GPULock) as character/asset generation. With
        IMAGE_PROVIDER=local, ref-conditioned frame renders auto-route to
        Qwen-Image-Edit-2509 via the manager's internal dispatch."""
        provider = os.getenv("IMAGE_PROVIDER", "wanx").lower()
        if provider == "local":
            from ...img_local.manager import ImageModelManager
            return ImageModelManager.get(model_config)
        if provider == "wanx":
            return WanxImageModel(model_config)
        raise ValueError(
            f"Unknown IMAGE_PROVIDER={provider!r}; expected 'wanx' or 'local'"
        )

    def generate_storyboard(self, script: Any) -> Any:
        """Generates images for all frames in the storyboard."""
        logger.info(f"Generating storyboard for script: {script.title}")
        
        total_frames = len(script.frames)
        for i, frame in enumerate(script.frames):
            logger.info(f"Generating frame {i+1}/{total_frames}: {frame.id}")
            
            # Skip if already completed (unless force regeneration is needed, but for now we skip)
            if frame.status == GenerationStatus.COMPLETED and frame.image_url:
                continue
                
            # Find scene for this frame
            scene = next((s for s in script.scenes if s.id == frame.scene_id), None)
            
            self.generate_frame(frame, script.characters, scene)
            
        return script

    def generate_frame(self, frame: StoryboardFrame, characters: List[Character], scene: Scene, ref_image_path: str = None, ref_image_paths: List[str] = None, prompt: str = None, batch_size: int = 1, size: str = None, model_name: str = None, force_image_refs: bool = False) -> StoryboardFrame:
        """Generates a storyboard frame image."""
        frame.status = GenerationStatus.PROCESSING
        
        # Default size for storyboard (landscape)
        effective_size = size or "1024*576"
        
        # Construct a rich prompt using character and scene details
        char_descriptions = []
        
        # Collect reference image paths from assets
        asset_ref_paths = []
        
        # If frontend provides explicit reference paths, use them directly
        # Otherwise, auto-collect from characters and scene
        use_frontend_refs = (ref_image_paths and len(ref_image_paths) > 0) or ref_image_path
        
        if use_frontend_refs:
            # Use only what frontend provided (already selected by user)
            if ref_image_paths:
                asset_ref_paths.extend(ref_image_paths)
            if ref_image_path:
                asset_ref_paths.append(ref_image_path)
            logger.info(f"[Storyboard] Using {len(asset_ref_paths)} frontend-provided reference images")
        else:
            # Auto-collect from characters and scene (fallback for batch generation)
            for char_id in frame.character_ids:
                char = next((c for c in characters if c.id == char_id), None)
                if char:
                    # Add character reference image - prioritize selected variant from ImageAsset
                    target_url = None
                    source = "none"
                    
                    # Priority 1: Use selected variant from three_view_asset
                    if char.three_view_asset and char.three_view_asset.selected_id:
                        selected_variant = next((v for v in char.three_view_asset.variants if v.id == char.three_view_asset.selected_id), None)
                        if selected_variant:
                            target_url = selected_variant.url
                            source = f"three_view_asset"
                    
                    # Priority 2: Use selected variant from full_body_asset
                    if not target_url and char.full_body_asset and char.full_body_asset.selected_id:
                        selected_variant = next((v for v in char.full_body_asset.variants if v.id == char.full_body_asset.selected_id), None)
                        if selected_variant:
                            target_url = selected_variant.url
                            source = f"full_body_asset"
                    
                    # Priority 3: Use selected variant from headshot_asset
                    if not target_url and char.headshot_asset and char.headshot_asset.selected_id:
                        selected_variant = next((v for v in char.headshot_asset.variants if v.id == char.headshot_asset.selected_id), None)
                        if selected_variant:
                            target_url = selected_variant.url
                            source = f"headshot_asset"
                    
                    # Priority 4: Fallback to legacy fields
                    if not target_url:
                        target_url = char.three_view_image_url or char.full_body_image_url or char.headshot_image_url or char.avatar_url or char.image_url
                        source = "legacy_fields"
                    
                    logger.info(f"[Storyboard] Character '{char.name}' reference: source={source}, url={target_url}")
                    
                    if target_url:
                        if is_object_key(target_url):
                            asset_ref_paths.append(target_url)
                        else:
                            potential_path = os.path.join("output", target_url)
                            if os.path.exists(potential_path):
                                asset_ref_paths.append(os.path.abspath(potential_path))
                            elif os.path.exists(target_url):
                                asset_ref_paths.append(os.path.abspath(target_url))
            
            # Add scene reference image
            scene_url = None
            if scene:
                if scene.image_asset and scene.image_asset.selected_id:
                    selected_variant = next((v for v in scene.image_asset.variants if v.id == scene.image_asset.selected_id), None)
                    if selected_variant:
                        scene_url = selected_variant.url
                if not scene_url:
                    scene_url = scene.image_url
                
                if scene_url:
                    if is_object_key(scene_url):
                        asset_ref_paths.append(scene_url)
                    else:
                        potential_path = os.path.join("output", scene_url)
                        if os.path.exists(potential_path):
                            asset_ref_paths.append(os.path.abspath(potential_path))
                        elif os.path.exists(scene_url):
                            asset_ref_paths.append(os.path.abspath(scene_url))
        
        # Collect character descriptions for prompt building
        for char_id in frame.character_ids:
            char = next((c for c in characters if c.id == char_id), None)
            if char:
                char_descriptions.append(f"{char.name} ({char.description})")
        
        char_text = ", ".join(char_descriptions)

        # Remove duplicates
        asset_ref_paths = list(set(asset_ref_paths))
        
        if not prompt:
            prompt = f"Storyboard Frame: {frame.action_description}. "
            if char_text:
                prompt += f"Characters: {char_text}. "
            if scene:
                prompt += f"Location: {scene.name}, {scene.description}. "
                
            prompt += f"Camera: {frame.camera_angle}"
            if frame.camera_movement:
                prompt += f", {frame.camera_movement}"
            prompt += "."
        else:
            # If prompt is provided by user/LLM, ensure character descriptions are still present for I2I consistency
            if char_text and char_text not in prompt:
                prompt = f"{prompt} Characters: {char_text}."
        
        # Store the optimized prompt
        frame.image_prompt = prompt
        
        # Initialize rendered_image_asset if not present
        if not frame.rendered_image_asset:
            frame.rendered_image_asset = ImageAsset(asset_id=frame.id, asset_type="storyboard_frame")

        try:
            import uuid
            
            for _ in range(batch_size):
                variant_id = str(uuid.uuid4())
                output_filename = f"{frame.id}_{variant_id}.png"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                
                # Local Qwen-Image-Edit-2509/Plus is currently too slow for
                # routine storyboard frames on 24GB cards. Keep scene-frame
                # generation on the fast T2I path by default; callers can
                # opt into the reference-conditioned edit path explicitly via
                # LOCAL_STORYBOARD_USE_IMAGE_REFS=1.
                effective_ref_paths = asset_ref_paths
                if (
                    os.getenv("IMAGE_PROVIDER", "wanx").lower() == "local"
                    and os.getenv("LOCAL_STORYBOARD_USE_IMAGE_REFS", "0").strip() != "1"
                    and not force_image_refs
                ):
                    effective_ref_paths = []

                logger.info(f"[Storyboard] Calling model.generate with {len(effective_ref_paths)} reference images using model {model_name or 'default'}")
                # Surface the frame id to the manager so the footer's
                # phase_label tells the user *which* frame is rendering.
                if hasattr(self.model, "set_generation_label"):
                    self.model.set_generation_label(f"frame {frame.id[:8]}")
                self.model.generate(prompt, output_path, ref_image_paths=effective_ref_paths, size=effective_size, model_name=model_name)
                
                # Store relative path for frontend serving
                rel_path = os.path.relpath(output_path, "output")
                
                # Create Variant
                variant = ImageVariant(
                    id=variant_id,
                    url=rel_path,
                    prompt=prompt,
                    created_at=time.time()
                )
                frame.rendered_image_asset.variants.append(variant)
                
                # Auto-select the latest one
                frame.rendered_image_asset.selected_id = variant_id
            
            # Sync legacy fields
            selected_variant = next((v for v in frame.rendered_image_asset.variants if v.id == frame.rendered_image_asset.selected_id), None)
            if selected_variant:
                frame.rendered_image_url = selected_variant.url
                frame.image_url = selected_variant.url
                
            frame.updated_at = time.time()
            frame.status = GenerationStatus.COMPLETED
            
            # Try uploading to OSS if configured - store Object Key (not full URL)
            try:
                from ...utils.oss_utils import OSSImageUploader
                uploader = OSSImageUploader()
                if uploader.is_configured:
                    # Upload the selected variant
                    if selected_variant:
                        # Construct local path from relative path
                        local_path = os.path.join("output", selected_variant.url)
                        if os.path.exists(local_path):
                            # Upload and get Object Key (not full URL)
                            object_key = uploader.upload_file(
                                local_path, 
                                sub_path=f"storyboard"
                            )
                            if object_key:
                                logger.info(f"Uploaded frame {frame.id} to OSS: {object_key}")
                                # Store Object Key (will be converted to signed URL on API response)
                                selected_variant.url = object_key
                                frame.rendered_image_url = object_key
                                frame.image_url = object_key
            except Exception as e:
                logger.error(f"Failed to upload frame {frame.id} to OSS: {e}")
                # Continue even if OSS upload fails
                
        except Exception as e:
            logger.error(f"Failed to generate frame {frame.id}: {e}")
            frame.status = GenerationStatus.FAILED
            
        return frame
