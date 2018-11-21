from PIL import Image
import os
import numpy as np
import pandas as pd
from tools.imtools import *

from pylab import *
import re

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
