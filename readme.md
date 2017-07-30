
# Rolland

This program automatically extracts single images of heraldry from book pages by detecting the grid and predicting missing values using regression.

<div align="center">
  <img src="https://github.com/anbn/heraldic/blob/master/images/figure_1.jpg"><br><br>
</div>

## Usage

The programs expects options for every page to be processed. For single pages, use `-s OPTION`, for double pages left `-l` and right `-r`, where option is either `ignore`, `small` (5,6 grid), `big` (7,6 grid) or `a:b`, where a and b define a grid of size (a,b). Note that grid sizes refers to grid crossings, an (a,b) grid contains (a+1)\*(b+1) heraldic signs. Option `-v` enables verbose mode to visualize the detected grid.

    python rolland.py -i IMAGE -o OUT-FOLDER -l OPTION -r OPTION -v

Example

    python rolland.py -i images/Aa.jpg -o out -l ignore -r big -v
