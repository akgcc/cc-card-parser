Scripts used to generate data for akgcc.github.io
To use (cc#4 in this example):
`python scraper.py -cc4clear`
Move invalid pictures from `./images-cc4clear/` to `./images-cc4clear/invalid`
`python parser.py -cc4clear`
To fix errors manually: create `data-cc4clear-fixes.json`, json in this file will be merged into data (upsert)
