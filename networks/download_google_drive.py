import os
import zipfile

url = "https://drive.google.com/uc?id=1RGg3h6Jgk128MBtqCHouvBB2Q4_vxjF_"
import gdown

output = 'model.zip'
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)
else:
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('..')
