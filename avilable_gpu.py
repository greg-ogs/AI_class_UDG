import tensorflow as tf
import torch

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
print('----------------------------------------------------------------')
if gpus:
    # Optionally, print details about each GPU
    for gpu in gpus:
        print('----------------------------------------------------------------')
        print(gpu)

print('----------------------------------------------------------------')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('----------------------------------------------------------------')

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {device_count}")

    # for i in range(device_count):
    #     gpu_properties = torch.cuda.get_device_properties(i)
    #     print(f"\nGPU {i}:")
    #     print(f"  Name: {gpu_properties.name}")
    #     print(f"  Capability: {gpu_properties.major}.{gpu_properties.minor}")
    #     print(f"  Total Memory: {gpu_properties.total_memory / 1024 ** 2} MB")
