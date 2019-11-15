import shutil, os

files = os.listdir('/media/cruigo/stuff/datasets/dogscats/train/')

count_cat = 0 # Number representing count of the cat image
count_dog = 0 # Number representing count of the dog image

for file in files:
	file = os.path.basename(file)
	if(file.startswith('cat') and file.endswith('jpg')):
		count_cat += 1
		shutil.copy('/media/cruigo/stuff/datasets/dogscats/train/' + file, '/media/cruigo/stuff/datasets/dogscats/train/cat/' + str(count_cat) + ".jpg")
	elif(file.startswith('dog') and file.endswith('jpg')):
		count_dog += 1
		shutil.copy('/media/cruigo/stuff/datasets/dogscats/train/' + file, '/media/cruigo/stuff/datasets/dogscats/train/dog/' + str(count_dog) + '.jpg')