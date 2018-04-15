# mostly taken from http://code.google.com/p/latexmath2png/

import os
import re
import sys
import tempfile
import subprocess


class Latex(object):
    INLINE = r'''
\documentclass[fleqn]{article} 
\usepackage{amssymb,amsmath,bm,color}
\usepackage[latin1]{inputenc}
\usepackage[active,textmath]{preview}
\begin{document}
\thispagestyle{empty}
\mathindent0cm
\parindent0cm 
%s
\end{document}
'''
    BLOCK = r'''
\documentclass[fleqn]{article} 
\usepackage{amssymb,amsmath,bm,color}
\usepackage[latin1]{inputenc}
\begin{document}
\thispagestyle{empty}
\mathindent0cm
\parindent0cm 
%s
\end{document}
'''

    def __init__(self, doc, dpi=120):
        self.doc = doc
        self.dpi = dpi

    def write(self):
        inline = bool(re.match('^\$[^$]*\$$', self.doc))
        if inline:
            TEX = self.INLINE
        else:
            TEX = self.BLOCK

        try:
            workdir = tempfile.gettempdir()
            fd, texfile = tempfile.mkstemp('.tex', 'eq', workdir, True)

            with os.fdopen(fd, 'w+') as f:
                f.write(TEX % self.doc)

            png, depth = self.convert_file(texfile, workdir)
            if not inline:
                depth = 0
            return png, depth

        finally:
            if os.path.exists(texfile):
                os.remove(texfile)

    def convert_file(self, infile, workdir):

        try:
            # Generate the DVI file
            cmd = 'latex -halt-on-error -output-directory %s %s'\
                    % (workdir, infile)

            p = subprocess.Popen(
                cmd,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            sout, serr = p.communicate()
            # Something bad happened, abort
            if p.returncode != 0:
                raise Exception('latex error', serr, sout)

            # Convert the DVI file to PNG's
            dvifile = infile.replace('.tex', '.dvi')
            pngfile = os.path.join(workdir, infile.replace('.tex', '.png'))
            cmd = "dvipng -T tight --depth -D %i -z 9 -bg Transparent -o %s %s" % (
                self.dpi, pngfile, dvifile)
            p = subprocess.Popen(
                cmd,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)

            sout, serr = p.communicate()
            if p.returncode != 0:
                raise Exception('dvipng error', serr)

            try:
                depth = int(re.search(r'\[1 depth=(-?\d+)\]', sout).group(1))
            except:
                depth = 0

            png = open(pngfile, 'rb').read()
            return png, depth

        finally:
            # Cleanup temporaries
            basefile = infile.replace('.tex', '')
            tempext = '.aux', '.dvi', '.log', '.png'
            for te in tempext:
                tempfile = basefile + te
                if os.path.exists(tempfile):
                    os.remove(tempfile)


__cache = {}


def tex2png(eq, **kwargs):
    if not eq in __cache:
        __cache[eq] = Latex(eq, **kwargs).write()
    return __cache[eq]


if __name__ == '__main__':
    src = sys.argv[1]
    print 'Equation is: %s' % src
    print Latex(src).write()
