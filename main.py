import numpy as np
import mp3_to_wav
import sh

def main():
    mp3_file = "mp3/music.mp3"
    wav_dir = "wav"
    mfcc_dir = "feature"
    sh.run("mkdir -p "+mfcc_dir)


    # modi_file = "feature/propriete.txt"
	# result_file = "result/result.txt"

    mp3_to_wav.run(mp3_file, wav_dir, mfcc_dir)





if __name__ == '__main__':
	main()
