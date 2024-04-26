import subprocess 

# method for capturing an image using the camera module

def take(filename):
	subprocess.run(["libcamera-still", "-o", filename + ".jpg"])

take("stillImage")
