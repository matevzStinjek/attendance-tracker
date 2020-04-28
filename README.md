# Attendance Tracker
## Facial detection and recognition with openCV in C++

### Purpose:
Recognise attendants and record their presence autonomously.

### Instructions:
  * Download the face database can be downloaded from [this link](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) into a designated folder, along with the contents of this repository.
  * Open the terminal and CD to the directory.
  * Run the following commands:
    * cmake .
    * make
    * ./presence <path to csv.ext> <path to faces directory> <path to face_cascade.xml>
    
### Usage:
Attendants are expected to face the front facing camera by the entrance of their designation and wait for a brief moment for the software to recognise them and mark them down as present. If this is their first time attending the software will save their faces to the folder faces and give them a new ID.
