License Plate Detection System
Description
AutoTrack is an automated License Plate Recognition (LPR) system designed to detect, identify, and log vehicle information from images. Utilizing computer vision and machine learning techniques, AutoTrack extracts license plate numbers, detects vehicle colors, and matches the information against a database of registered vehicles. The system also sends email notifications with relevant data and maintains a MySQL database of vehicle details.

Features
License Plate Detection: Automatically detects and reads license plates from uploaded images using OpenCV and Tesseract OCR.
Vehicle Color Detection: Identifies the dominant color of the vehicle using KMeans clustering.
Database Management: Stores and retrieves vehicle information in a MySQL database.
Email Notifications: Sends email notifications with the details of detected vehicles.
Theming: Switch between light and dark mode for a better user interface experience.
Data Matching: Compares detected license plates with a CSV file to retrieve additional vehicle details.


Technologies Used
Python: Core programming language for the application.
OpenCV: Library for computer vision tasks.
Tesseract OCR: Optical character recognition engine.
Tkinter: Python GUI library for building the desktop interface.
MySQL: Database management system.
Pandas: Data manipulation and analysis library.
Sklearn: Machine learning library for color detection.
Smtplib: Library for sending emails.
PandasTable: Display and manage table data in the GUI.


Installation
Clone the repository:

Install dependencies:

pip install opencv-python imutils numpy pytesseract pillow mysql-connector-python pandas pandastable scikit-learn

Download Tesseract OCR:
Note the installation path and update the pytesseract.pytesseract.tesseract_cmd variable in the script accordingly.

Prepare the Database:

Ensure you have MySQL installed and running.
Create a database named vehicle_info and update the connection details in the script if necessary.
Prepare the Excel/CSV files:

Ensure vish.xlsx and vehicle_data.csv files are present in the project directory with the necessary vehicle data.

Run the application:

python license_plate_detection.py

Notes
Ensure that the paths and database connection details are correctly configured.
Modify email settings (sender, receiver, and password) before using the email functionality.
Ensure the required Excel/CSV files are present and correctly formatted.

License
This project is licensed under the MIT License.