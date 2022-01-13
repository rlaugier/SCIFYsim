from pathlib import Path

import logging

logit = logging.getLogger(__name__)
def set_logging_level(level=logging.WARNING):
    logit.setLevel(level=level)
    logit.info("Setting the logging level to %s"%(level))

def convert_config_file(file_in, path_out, new_name=None, test=True):
    """
    A simple routine to convert .prm config file to .ini config files
    
    **Paramters**
    
    * file_in    : the complet path of the file to convert
    * path_out   : the destination directory (not including file name)
    * new_name   : the new name to give the file (if None: change only extension)
    * test       : If True, only print the result
    
    *example* : ``convert_config_file("../../geniesim/examples/tau_Boo_89m.prm", "examples/", test=False)``
    """
    if test:
        print("Test run: no file will be written")
    file_in = Path(file_in)
    path_out = Path(path_out)
    thefile = open(file_in, "r",errors="replace")
    thetext = thefile.readlines()
    thefile.close()
    nalign = 30

    lines = []
    for line in thetext:
        #print(line[:50])
        linesplit = line.split(";")
        if linesplit[0] is "": # If line started with ";", then it si a section name
            section_name = linesplit[1].strip()
            sect_line = "["+section_name+"]\n"
            if test : print(sect_line)
            lines.append(sect_line)
        else : # Parse
            valueraw = linesplit[0].strip()
            value = valueraw.replace("D", "e")

            #print("data", value)
            textclean = linesplit[1].strip().split(" ")
            name = textclean[0]
            important = name+" = "+value
            skip = nalign - len(important)
            if skip <= 0:
                skip = 1
            new_line = important+skip*" "+"# "+linesplit[1].strip()+"\n"
            if test: print(new_line)
            lines.append(new_line)
    
    if new_name is None:
        splitname = (file_in.name).split(".")[:-1]
        splitname.append("ini")
        outname = ".".join(splitname)
    else :
        outname = new_name
    file_out = path_out.absolute()/outname
    print("Destination", file_out)
    if not test:
        theoutfile = open(file_out, "w+")
        theoutfile.writelines(lines)
        theoutfile.close()
    else:
        print("No file written")
    #print("name", name)
    #print("The nb of textsplit", len(textsplit))