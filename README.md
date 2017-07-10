### Notes
I will be updating this repo to handle some platform compatibility issues soon

## Abstract
In the initial analysis of a data set it is useful to gather informative 
summaries. This includes evaluating the available fields, by finding unique 
counts or by calculating summary statistics such as averages for numerical 
fields. These summaries help in understanding what is in the data itself, the 
underlying quality, and illuminate potential paths for further exploration. In 
structured data, this a straightforward task, but for unstructured text, 
different types of summaries are needed. Some useful examples for text data 
include a count of the number of documents in which a term occurs, and the 
number of times a term occurs in a document. Since vocabulary terms often have 
variant forms, e.g. “performs” and “performing”, it is useful to pre-process 
and combine these forms before computing distributions. Oftentimes, we want to 
look at sequences of words, for example we may want to count the number of 
times “data science” occurs, and not just “data” and “science”. We will use the 
pandas Python Data Analysis Library and the Natural Language Toolkit (NLTK) to 
process a data set of job descriptions posted by employers in the United 
States, and look at the difference in vocabularies across different job 
segments.

### Prerequisites
For "Vocabulary Analysis of Job Descriptions", the tutorial will be done using 
Jupyter notebooks, so it would be good to have a Jupyter notebook server 
running. The Anaconda installer comes with most of the libraries that the 
tutorial will use: numpy, pandas, matplotlib, scikit-learn, and NLTK. Although 
NLTK is installed with Anaconda, the data may not be, so attendees should 
install at least the "book" collection of NLTK data (general NLTK data 
installation instructions). The only additional library that the tutorial will 
use is wordcloud [word_cloud](https://github.com/amueller/word_cloud) which can be installed by following the instructions 
on the linked github page.
**Start with Setup.ipynb**

## Docker Notes
```bash
docker build -t pydata-vocab-ana .
docker run -p 8889:8888 --name vocabana pydata-vocab-ana
```
To avoid OOM killer, shutdown each notebook when you are finished before continuing to the next.
