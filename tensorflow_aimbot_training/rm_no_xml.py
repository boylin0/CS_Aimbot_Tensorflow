import os


dir = 'collected_images'

filelist = os.listdir(dir)
filelist.sort()
os.chdir(dir)
for filename in filelist:
  file_name ,file_extension = os.path.splitext(filename)
  if file_extension == '.jpg':
    if not os.path.isfile(file_name+'.xml'):
      os.remove(filename)
      print('Remove: {}'.format(filename))
