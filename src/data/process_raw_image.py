from importlib.resources import path
from PIL import Image

# function for resizing and cropping to 64x64
def process_raw_image(input_imgpath: path, output_imgpath: path, output_imgname: str, imgsize: int=64) -> object:
    '''Input a path to a png-image and output a cropped and resized square image saved in a desired folder.'''
    
    im = Image.open(input_imgpath)

    # Get size
    x, y = im.size

    # New sizes
    yNew = imgsize
    xNew = yNew 
    
    # First, set right size
    if x > y:
        # Y is smallest, figure out relation to 256
        xNew = round(x * imgsize / y)
    else:
        yNew = round(y * imgsize / x)

    # resize
    im = im.resize((int(xNew), int(yNew)), Image.ANTIALIAS)

    # crop
    im = im.crop(((int(xNew) - imgsize)/2, (int(yNew) - imgsize)/2, (int(xNew) + imgsize)/2, (int(yNew) + imgsize)/2))

    # im = im.resize((imgsize, imgsize), Image.ANTIALIAS)
    print(im.size)
    print(f"Saving image in {output_imgpath}")
    im.save(output_imgpath+f"{output_imgname}.png")