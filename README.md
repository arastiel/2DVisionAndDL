2D Vision and Deep learning WiSe 2020-2021
==========================================

This is your submission repository
Each time there is created a new submission folder `assignmentXX` in your repository. 
You'll find there the assignment PDF. Also your solution has to be placed in this folder. 
Please do not create these folders by yourself to avoid git conflicts.

## Assignments 

### Checkout
Normally on Monday after the lecture, the new assignment will be uploaded and pushed to your repository. 
Then you can update your repository via git. 

### Submission
Your submission is done via a git commit with a meaningful commit message and via pushing your results to the gitlab system. 
Please be aware that commiting your solution without pushing it to the server
is not counted. You can verify your submission state online at the gitlab server.

### Submission structure
Each assignment gets an individual folder. Put your exercises in these folders. Each 
exercise in a separate with filename `exercise_XX.py`. These files are the entry points 
for each exercise (i.e. the have the main/running code). 
You can create common methods or classes inside each assignment folder as module file. 

### Data 
Large data have to be placed in data folder. There will a module `data.py` in the assignment
folder with the paths to the data. The data is not included in the repository, you
have to download them from lms.uni-mainz.de.  

### Working Groups
You can work in groups of 4 or 5 persons. Use the git system to synchronize each of your indivdual work. 
Be aware of merge conflicts and resolve them. 

### Correction and Evaluation 
After submission deadline the current state of your repository is downloaded by practical group assistant. 
Not the last version before submission deadline is used for evaluation. All changes after submission will not
be evaluated. 
Points and remarks are uploaded afterwards and you'll be able to see them via a new local fetch and merge of your repository. 
Your points are saved in a file called `points/your_points.txt` and maximum points in `points/max_points.txt`. 
Do not change or edit these files!

You can get an overview of your points by starting `points/visualize_points.py`, which reads both files. 
This requires the python package PyQt5

### Comments are not optional
Please document your code. This helps to catch up your idea. If the idea is not clear and no 
comment is available, you won't get all points.
