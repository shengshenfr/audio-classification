import numpy as np
import mp3_to_wav
import sh



def main():
    mp3_file = "mp3/sound.mp3"
    wav_dir = "wav"
    mfcc_dir = "feature"
    sh.run("mkdir -p "+mfcc_dir)


    mp3_to_wav.run(mp3_file, wav_dir)








if __name__ == '__main__':
	main()
