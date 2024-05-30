#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace std;
using namespace cv;
// Function to apply Gaussian blur to an image chunk
void applyGaussianBlur(Mat& chunk, int ksize, double sigmaX, double sigmaY) {
    GaussianBlur(chunk, chunk, Size(ksize, ksize), sigmaX, sigmaY);// Example using OpenCV's GaussianBlur function
}

// Function to perform edge detection on an image chunk
void applyEdgeDetection(Mat& chunk) {
    Canny(chunk, chunk, 100, 200); // Example using OpenCV's Canny edge detection
}

// Function to rotate an image chunk
void rotateImage(Mat& chunk, double angle) {
    Point2f center(chunk.cols / 2.0, chunk.rows / 2.0);
    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(chunk, chunk, rotationMatrix, chunk.size());
}

// Function to scale an image chunk
void scaleImage(Mat& chunk, double scaleX, double scaleY) {
    resize(chunk, chunk, Size(), scaleX, scaleY);
}

// Function to apply histogram equalization to an image chunk
void applyHistogramEqualization(Mat& chunk) {
    cvtColor(chunk, chunk, COLOR_BGR2GRAY); // Convert the chunk to grayscale
    equalizeHist(chunk, chunk); // Apply histogram equalization
}

// Function to convert color space of an image chunk
void convertColorSpace(Mat& chunk, int conversionCode) {
    cvtColor(chunk, chunk, conversionCode); // Convert the color space
}

// Function to apply global thresholding to an image chunk
void applyGlobalThresholding(Mat& chunk) {
    cvtColor(chunk, chunk, COLOR_BGR2GRAY); // Convert the chunk to grayscale
    threshold(chunk, chunk, 127, 255, THRESH_BINARY); // Apply global thresholding
}

// Function to apply local thresholding to an image chunk
void applyLocalThresholding(Mat& chunk, int blockSize, int constant) {
    cvtColor(chunk, chunk, COLOR_BGR2GRAY); // Convert the chunk to grayscale
    adaptiveThreshold(chunk, chunk, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, constant); // Apply local thresholding
}

// Function to apply median filter to an image chunk
void applyMedianFilter(Mat& chunk, int kernelSize) {
    medianBlur(chunk, chunk, kernelSize); // Apply median filter with the specified kernel size
}


// Function to perform the selected image processing operation
void performOperation(Mat& localChunk, int operationType, string outputFileName) {
    double startTime = MPI_Wtime();
    string filterName;
    switch (operationType) {
    case 1:
        // Apply Gaussian blur
        int ksize;
        double sigmaX, sigmaY;
        cout << "Enter the size (odd number): ";
        cin >> ksize;
        cout << "Enter the value of sigma X: ";
        cin >> sigmaX;
        cout << "Enter the value of sigma Y: ";
        cin >> sigmaY;
        applyGaussianBlur(localChunk, ksize, sigmaX, sigmaY);
        filterName = "Gaussian Blur";
        break;
    case 2:
        // Apply edge detection
        applyEdgeDetection(localChunk);
        filterName = "Edge Detection";
        break;
    case 3:
        // Rotate the image
        double angle;
        cout << "Enter the angle for image rotation: ";
        cin >> angle;
        rotateImage(localChunk, angle); // Rotate by the specified angle
        filterName = "Image Rotation";
        break;

    case 4:
        // Scale the image
        double scaleX, scaleY;
        cout << "Enter the scaling factor for the X axis (less than 1): ";
        cin >> scaleX;
        cout << "Enter the scaling factor for the Y axis (less than 1): ";
        cin >> scaleY;
        scaleImage(localChunk, scaleX, scaleY);
        filterName = "Image Scaling";
        break;
    case 5:
        // Apply histogram equalization
        applyHistogramEqualization(localChunk);
        filterName = "Histogram Equalization";
        break;
    case 6:
        // Convert color space to LAB
        convertColorSpace(localChunk, COLOR_BGR2Lab);
        filterName = "Color Space Conversion";
        break;
    case 7:
        // Apply global thresholding
        applyGlobalThresholding(localChunk);
        filterName = "Global Thresholding";
        break;
    case 8:
        // Apply local thresholding
        int blockSize, constant;
        cout << "Enter the block size for local thresholding: ";
        cin >> blockSize;
        cout << "Enter the constant for local thresholding: ";
        cin >> constant;
        applyLocalThresholding(localChunk, blockSize, constant);
        filterName = "Local Thresholding";
        break;
    case 9:
        // Apply image compression (placeholder)
        cout << "Image compression operation not implemented yet." << endl;
        return;
    case 10:
        // Apply median filter
        int kernelSize;
        cout << "Enter the kernel size for the median filter: ";
        cin >> kernelSize;
        applyMedianFilter(localChunk, kernelSize);
        filterName = "Median Filter";
        break;
    default:
        cerr << "Invalid operation type." << endl;
        return;
    }
    double endTime = MPI_Wtime();
    double duration = endTime - startTime;
    cout << filterName << " completed successfully. Time taken: " << duration << " seconds" << endl;
    cout << "blurred image saved as" << outputFileName << endl;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "Welcome to Parallel Image Processing with MPI" << endl;
        cout << "Please choose an image processing operation:" << endl;
        cout << "01- Gaussian Blur" << endl;
        cout << "02- Edge Detection" << endl;
        cout << "03- Image Rotation" << endl;
        cout << "04- Image Scaling" << endl;
        cout << "05- Histogram Equalization" << endl;
        cout << "06- Color Space Conversion" << endl;
        cout << "07- Global Thresholding" << endl;
        cout << "08- Local Thresholding" << endl;
        cout << "09- Image Compression" << endl;
        cout << "10- Median Filter" << endl;
        int choice;
        cout << "Enter your choice: ";
        cin >> choice;

        
        // Broadcast the choice to all processes
        
        // Load the image on process 0
        Mat image;
        string inputFileName;
        cout << "Please enter the filename of the input image (e.g., input.jpg): ";
        cin >> inputFileName;
        string outputFileName;
        cout << "Please enter the filename for the output blurred image (e.g., output.jpg): ";
        cin >> outputFileName;
        string inputImagePath = "D:/parallel/parallel/parallel/" + inputFileName;  // Concatenating directory path with filename
        image = imread(inputImagePath, IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Failed to load image." << endl;
            MPI_Finalize();
            return -1;


        // Broadcast the image dimensions to all processes
        int rows, cols;
        if (rank == 0) {
            rows = image.rows;
            cols = image.cols;
        }
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate the size of each chunk
        int sup_row = rows / size;

        // Scatter image data to different processes
        Mat localChunk(sup_row, cols, CV_8UC3);
        MPI_Scatter(image.data, sup_row * cols * 3, MPI_BYTE, localChunk.data, sup_row * cols * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

        // Choose the operation to perform
        int operationType = 0; // Change this value to switch between operations (0 for median filter, 1 for local thresholding)
        performOperation(localChunk, operationType,"");

        // Gather processed chunks from all processes on process 0
        Mat resultImage;
        if (rank == 0) {
            resultImage = Mat::zeros(image.size(), image.type());
        }
        MPI_Gather(localChunk.data, sup_row * cols * 3, MPI_BYTE, resultImage.data, sup_row * cols * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

        // Save the result image on process 0
        if (rank == 0) {
            string outputImagePath = "D:/parallel/parallel/parallel/" + outputFileName; // Concatenating directory path with filename
            imwrite(outputImagePath, resultImage);
            cout << "Blurred image saved as " << outputImagePath << endl;

            cout << "Thank you for using Parallel Image Processing with MPI." << endl;
        }

        MPI_Finalize();
        return 0;
     /*  Mat image;
        string inputFileName;
        cout << "Please enter the filename of the input image (e.g., input.jpg): ";
        cin >> inputFileName;
        string outputFileName;
        cout << "Please enter the filename for the output blurred image (e.g., output.jpg): ";
        cin >> outputFileName;
        string inputImagePath = "D:/parallel/parallel/parallel/" + inputFileName;  // Concatenating directory path with filename
        image = imread(inputImagePath, IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Failed to load image." << endl;
            MPI_Finalize();
            return -1;

        }

        // Broadcast the image dimensions to all processes
        int rows = image.rows;
        int cols = image.cols;
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate the size of each chunk
        int sup_row = rows / size;

        // Scatter image data to different processes
        Mat localChunk(sup_row, cols, CV_8UC3);
        MPI_Scatter(image.data, sup_row * cols * 3, MPI_BYTE, localChunk.data, sup_row * cols * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

        // Perform the selected image processing operation
        performOperation(localChunk, choice);

        // Gather processed chunks from all processes on process 0
        Mat resultImage = Mat::zeros(image.size(), image.type());
        MPI_Gather(localChunk.data, sup_row * cols * 3, MPI_BYTE, resultImage.data, sup_row * cols * 3, MPI_BYTE, 0, MPI_COMM_WORLD);
        // Save the result image on process 0
        //string outputFileName;
        //cout << "Please enter the filename for the output blurred image (e.g., output.jpg): ";
        //cin >> outputFileName;
        string outputImagePath = "D:/parallel/parallel/parallel/" + outputFileName; // Concatenating directory path with filename
        imwrite(outputImagePath, resultImage);
        cout << "Blurred image saved as " << outputImagePath << endl;

        cout << "Thank you for using Parallel Image Processing with MPI." << endl;
    }
 
    MPI_Finalize();
    return 0;*/
}
