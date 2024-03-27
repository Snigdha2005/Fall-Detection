'''execute extract dataset to a folder'''
import tarfile

tgz_file_path = 'publicFallDetector201307 (1).tgz'

extract_path = 'temp_extract'

with tarfile.open(tgz_file_path, 'r:gz') as tar:
    tar.extractall(extract_path)

