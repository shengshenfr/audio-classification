import numpy as np
import mp3_to_wav
import sh
import sample


def main():
    mp3_file = "mp3/sound.mp3"
    wav_dir = "wav"
    mfcc_dir = "feature"
    sh.run("mkdir -p "+mfcc_dir)


    sample_file = "sample/sample.txt"
    result_file = "result/result.txt"

    mp3_to_wav.run(mp3_file, wav_dir, mfcc_dir)

    sample.read_sample_feature(mfcc_dir,sample_file)






if __name__ == '__main__':
	main()
