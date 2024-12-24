# Foundation-Shade-Detection

<h1>Introduction</h1>
<p>This project focuses on detecting and analyzing skin color from a user's image using a machine learning model. The users are instructed to take an image in a well evenly lit environment holding a white sheet of paper. The image is used to create a mask of skin pixels and that is used to extract dominant skin tones using k-means clustering. It further finds the closest skin tone for the user by finding the pixels of the white paper and normalising the skin color with the help of the difference of the color of the paper from true white. It returns a final skin tone for the user which is then classified into one of the 10 skin tones using Random Forest classification.</p>

<b>Key Features:</b>
<ul>
<li>Skin tone extraction using HSV masks.</li>
<li>Color clustering with KMeans for dominant tone identification.</li>
<li>White pixels extraction using HSV mask</li>
<li>Normalizing the skin tone using the white difference </li>
<li>Predictive modeling using Random Forest with optimal hyperparameters.</li>
</ul>


<h1> Installation Instructions </h1>

Clone the Repository: <br>
git clone https://github.com/izbaatif/Foundation-Shade-Detection <br>

cd to the path on local environment in terminal <br>
cd project path <br>

Install Dependancies by:<br>
pip install -r requirements.txt <br>

<h1>Usage Guide</h1>

cd to the project src directory and run the code<br>
python run.py<br>

The project will run on localhost:3000/


<h1>Visuals</h1>
