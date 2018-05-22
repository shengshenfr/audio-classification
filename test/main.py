import numpy as np
import mp3_to_wav
import sh


def main():
    mp3_dir = "mp3"
    # wav_dir = "wav"
    feature_dir = "feature"
    sh.run("mkdir -p "+mfcc_dir)


    mp3_to_wav.run(mp3_file, wav_dir)








if __name__ == '__main__':
	main()
