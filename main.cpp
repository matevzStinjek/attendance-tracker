#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <sys/stat.h>

using namespace cv;
using namespace cv::face;
using namespace std;

CascadeClassifier cascade;
string fn_csv;
string result;
string path;
int numStudents;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, int& numStudents, char separator = ';') {
    
    ifstream file(filename.c_str(), ifstream::in);
    
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    
    string line, path, classlabel;
    
    while (getline(file, line)) {
        
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        
        if(stoi(classlabel) > numStudents)
            numStudents = stoi(classlabel);
        
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
    
    numStudents++;
}

static void filter(Mat& in, Rect roi){
    in = in(roi);
    resize(in, in, Size(92,112));
}

static int mostCommonElement(vector<int> a){
    int count = 1, current;
    int popular = a.at(0);
    int temp = 0;
    for (int i = 0; i < a.size() - 1; i++){
        temp = a[i];
        current = 0;
        for (int j = 1; j < a.size(); j++){
            if (temp == a[j])
                current++;
        }
        if (current > count){
            popular = temp;
            count = current;
        }
    }
    return popular;
}

static void addToPresence(vector<int>& presence, int label){
    for(int i = 0; i < presence.size(); i++){
        if(presence.at(i) == label)
            return;
    }
    presence.push_back(label);
}

static void judge(int& c, vector<int>& confidenceValues, vector<int>& guesses, vector<Mat>& newEntry, vector<int>& presence){
    c = 0;
    
    int confidence = confidenceValues.at(0);
    int confidenceLabel = guesses.at(0);
    
    for(int i = 1; i < confidenceValues.size(); i++){
        if(confidenceValues.at(i) < confidence){
            confidence = confidenceValues.at(i);
            confidenceLabel = guesses.at(i);
        }
    }
    
    if(confidence < 650 || (mostCommonElement(guesses) == confidenceLabel && confidence < 1000)){
        // MATCH FOUND
        addToPresence(presence, confidenceLabel);
        result = format("Welcome back student ID %d. Next please!", confidenceLabel);
        return;
    }
    
    // MATCH NOT FOUND, ENTER TO DATABASE
    addToPresence(presence, numStudents);
    result = format("Welcome to openCV 101! Your ID is %d", numStudents++);
    cout << format("saving to %s/s%d", path.c_str(), numStudents) << endl;
    
    string folder = format("%s/s%d", path.c_str(), numStudents);
    
//    char *mkfolder = nullptr;
//    sprintf(mkfolder, "%s/s%d", path.c_str(), numStudents);
//    cout << mkfolder << endl;
//    mkdir(folder, S_IRWXU);
    
    ofstream outfile;
    outfile.open(fn_csv, ios_base::app);
    
    for(int i = 0; i < newEntry.size(); i++){
        string imgpath = format("%s/%d.pgm", folder.c_str(), i+1);
        imwrite(imgpath, newEntry.at(i));
        outfile << format("%s;%d", imgpath.c_str(), numStudents-1) << endl;
    }
    
}

int main(int argc, const char *argv[]) {
    
    if (argc < 4){
        cout << "To run the program, you need the following paths:" << endl;
        cout << "<csv.ext> <faces directory> <face_cascade.xml>" << endl;
        cout << "The face database can be downloaded from http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html" << endl;
    }
    
    //////
    // LOAD FROM FILE
    
    fn_csv = string(argv[1]);
    
    vector<Mat> images;
    vector<int> labels;
    
    numStudents = 0;
    path = string(argv[2]);
    
    try {
        read_csv(fn_csv, images, labels, numStudents);
    } catch (Exception& e) { exit(1); }
    
    vector<int> presence;
    
    //////
    // TRAIN
    
    Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();
    model->train(images, labels);
    
    //////
    // DETECT FACE
    
    if( !cascade.load(string(argv[3])) ) {
        cerr << "Could not load classifier cascade" << endl;
        return -1;
    }
    
    VideoCapture camera(0);
    
    if (!camera.isOpened()) { 
        cerr << "Unable to access camera" << endl;
        return -1;
    }
    
    Mat frame, gray;
    vector<Rect> faces;
    vector<Mat> newEntry;
    
    vector<int> guesses;
    vector<int> confidenceValues;
    
    int c = 0;
    
    bool skip = false;
    
    while (true) {
        
        camera.read(frame);
        
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        faces.clear();
        cascade.detectMultiScale(gray, faces,
                                 1.1, 3, CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH | CASCADE_SCALE_IMAGE,
                                 Size(280, 280), Size(400, 400) );
        
        bool face = faces.size() == 1;
        if(faces.size() == 0)
            skip = false;
        
        if(skip){
            putText(frame, result, Point(80,550), FONT_HERSHEY_DUPLEX, 1.7, Scalar(0,255,0));
        } else if(face){
            
            c++;
            
            Rect r = faces[0];
            
            rectangle(frame,
                      Point(round(r.x), round(r.y)),
                      Point(round((r.x + r.width-1)), round((r.y + r.height-1))),
                      Scalar(255, 255, 255), 3, 8, 0);
            
            Point a = Point(round(r.x), round(r.y) - (r.y + r.height) * 0.1);
            Point b = Point(round((r.x + r.width-1)), round((r.y + r.height-1)) * 1.1);
            
            Rect param;
            param.x = a.x;
            param.y = a.y;
            param.height = b.y - a.y;
            param.width = b.x - a.x;
            
            if(c % 3 == 0){
                filter(gray, param);
                int predictedLabel = -1;
                double confidence = 0.0;
                model->predict(gray, predictedLabel, confidence);
                cout << format("predict: %d | confidence: %f", predictedLabel, confidence) << endl;
                newEntry.push_back(gray.clone());
                guesses.push_back(predictedLabel);
                confidenceValues.push_back(confidence);
            }
        } else if(faces.size() > 1)
            putText(frame, "One at a time, please!", Point(280,420), FONT_HERSHEY_DUPLEX, 2.0, Scalar(0,0,255));
        else
            putText(frame, "Please step in front of the screen!", Point(80,550), FONT_HERSHEY_DUPLEX, 2.0, Scalar(255,0,0));
        
        if(c > 30){
            skip = true;
            judge(c, confidenceValues, guesses, newEntry, presence);
            confidenceValues.clear();
            guesses.clear();
            newEntry.clear();
        }
        
        imshow("WINDOW", frame);
        if (waitKey(10) >= 0)
            break;
    }
    
    cout << "Prisotni: ";
    for (int i = 0; i < presence.size(); i++)
        cout << presence.at(i) << " ";
    cout << endl;
    
    return 0;
}
