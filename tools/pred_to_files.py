
import argparse
import glob
import os

def get_parser():
    parser = argparse.ArgumentParser(description="Tool for split frames to files from detection export file.")
    parser.add_argument("--input-file", "-i", help="Path to detection export file.", required=True)
    parser.add_argument("--output-dir", "-o", help="Output directory where frame files will be save.", required=True)
    return parser

def split_frames_to_files(input_file, output_dir):
    fmain = open(input_file, "r")
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as exc:
            print(f"Can not create destination directory {os.path.abspath(output_dir)}!")
            print(exc.strerror)
            exit(exc.errno)
       
    output_dir = os.path.abspath(output_dir)
    
    last_frame_id = 0
    ffile = None
    for line in fmain:
        sline = line.split("\t")
        frame_id = int(sline[0])   
        object_class = int(sline[1])
        if object_class == 0:
            object_class = "people"
        if object_class == 32:
            object_class = "ball"
        sline[6] = sline[6].replace("\n", "")
        score = sline[6] #2 or 6
        boxX = sline[2] #3 or 2
        boxY = sline[3] #4 or 3
        boxWidth = sline[4] #5 or 4
        boxHeight = sline[5] #6 or 5
        if frame_id != last_frame_id:
            if ffile != None and not ffile.closed:
                ffile.close()
            ffile = open(f"{output_dir}/{frame_id}.txt", "w")
            ffile.write(f"{object_class} {score} {boxX} {boxY} {boxWidth} {boxHeight}\n")
            last_frame_id = frame_id
        else:
            ffile.write(f"{object_class} {score} {boxX} {boxY} {boxWidth} {boxHeight}\n")
            
    fmain.close()

if __name__ == "__main__":
    args = get_parser().parse_args()

    assert os.path.isfile(args.input_file)
    if not args.input_file or not args.output_dir:
        print("Provide required arguments!")
        exit(1)    
                
    split_frames_to_files(args.input_file, args.output_dir)


    