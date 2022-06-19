# coding: utf-8
import os
import sys
import datetime
import subprocess
import pandas as pd
import argparse


def vprint(*print_args):
    """Prints the passed arguments based on the verbosity level at which
    the program is run.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=int, default=0)
    parser.add_argument('--label', type=str)
    parser.add_argument('--base', type=str)
    parser.add_argument('--chrom', type=str)
    parser.add_argument('--sp', type=int, default=0)
    parser.add_argument('--ep', type=int, default=1000)
    parser.add_argument('--onlycovs', type=bool, default=False)
    parser.add_argument('--gene', type=str)
    parser.add_argument('--win', type=int)
    parser.add_argument('--win_size', type=int, default=50)
    parser.add_argument('--flanking', type=int, default=2500)
    
    args = parser.parse_args()

    if args.v:
        print(*print_args)
    else:
        return

def generate_markdown_page(content, md_fname, fmode='w'):
    """Generates markdown content from a list of different content and
    then appends it to the specified markdown file.

    Parameters
    ----------
    content : dict
        Dictionary of content to be added to the markdown file. The keys
        are the files and the values are the content types.
        Types:
            I/i = image,
            P/p = paragraph, 
            T/t = table,
            H{1:5}+/h{1:5}+ = heading   
    md_fname : str
        File path to save the markdown file as. 
    fmode: str, optional
        Mode to open file in ('a', 'w'), by default 'w'.
    """

    wd = '/'.join(md_fname.split('/')[:-1])

    # Current date and time
    now = datetime.datetime.now()
    fmt = "%d/%m/%Y %H:%M"
    day = now.strftime(fmt)
    md_text = ''
    if (not os.path.isfile(md_fname)) or fmode == 'w':
        md_text += '[Summary](../Summary.html) \n\n'
    
    md_text += '{}\n\n'.format(day)
    
    for c, t in content.items():
        md_c = ''
        # Image
        if t == 'I' or t == 'i':
            c = os.path.relpath(c, wd)
            # if c.endswith('svg'):
            #     svg2png(open(c, 'r').read(), c.replace('svg', 'png'))
            #     c.replace('svg', 'png')
            imfile = c.split('/')[-1]
            imname = ''.join(imfile.split('.')[:-1])
            # md_c = '![{}]({})'.format(imname, c)
            md_c = '![]({})'.format(c)
        
        # Text paragraph
        elif t == 'P' or t == 'p': 
            c_lines = c.split('\n')
            if len(c_lines) == 1:
                md_c = c
            else:
                for l in c_lines:
                    md_c += '*{}\n'.format(l)
            
        # Heading
        elif t.startswith('H') or t.startswith('h'):
            hlevel = int(t[1])
            md_c = '\n{} {}\n'.format('#'*hlevel, c)
        
        # Table
        elif t == 'T' or t == 't':
            if isinstance(c, str):
                df = pd.read_csv(c)
                md_c = df.to_markdown()
            else:
                md_c = c.to_markdown()

        # Links
        elif t == 'L' or t == 'l':
            c = os.path.relpath(c, wd)
            lfile = c.split('/')[-1]
            lname = ''.join(lfile.split('.')[:-1])
            md_c = '* [{}]({})'.format(lname, c)

        md_text += md_c + '\n\n'
    
    with open(md_fname, fmode) as f:
        f.write(md_text)
    subprocess.run(['pandoc', 
        '-s', md_fname,
        '-c', '/home/upamanyu/GWASOnSteroids/LabNotebook/mvp.css', 
        '-o', md_fname.replace('md', 'html')])

    