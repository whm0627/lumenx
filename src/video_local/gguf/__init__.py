"""Generic GGUF infrastructure — model-agnostic. Reader parses the
file format; dequant kernels handle each quant type's bit layout;
GGUFLinear is a drop-in nn.Linear replacement that holds quantized
bytes and dequantizes per forward call."""
