# HAR (Human activity recognition)
The objective of project is to analyze dataset of sensor based data of human activities.
By using human-engineered 561 features of 128 reading, our goal is to predict one of the six activities that a user perform.

### INTRODUCTION:

Whilst camera-based data collection systems are limited to a predefined range, mobile sensor devices have unlimited range. Smartphones and wearables have a high autonomy and are thus capable of collecting and analysing data over longer time periods. Depending on the activity acceleration, orientation and other movement related signals are used. These can be enriched by environmental parameters such as temperature, noise or humidity.
Many successful applications of HAR use machine learning for pattern recognition due to its ability to generalize. To capture the temporal dependencies between the data recurrent neural networks (RNN) as well as long short-term memory (LSTM) are often applied.
The objective of the undertaking was to identify the percentage of waiting (standing), sitting, walking and jogging. This project comes under computer-vision statistical-based methodologies, syntactic methodologies and descriptions-based methodologies for hierarchical recognition are examined. Whilist camera-based data collection system are limited to a predefined range, mobile sensor devices have unlimited range.
Two of such sensors are: accelerometer (measures acceleration) and gyroscope (measures angular velocity), captures 3-dimensional linear acceleration and angular velocity
This is multi-class classification problem

![image](https://user-images.githubusercontent.com/70462853/127045305-e45924d0-4362-4393-ab1f-6089d4052015.png)

 

			                   Fig.1 steps to carry project
					   
![image](https://user-images.githubusercontent.com/70462853/127045400-aea28b8c-6c84-4e33-a8ed-142c09dd04cd.png)

			Fig.2 categories of deep learning in sensors based human activity recognition

### PROBLEM STATEMENT: 

To analyse a dataset of smartphone sensor data of human activities of about 128 participants and try to analyse the same and draw insights and predict the activity using Machine Learning. We also try to detect if we could identify the participants from their walking styles and try to draw additional insights. By using either human-engineered 561 features of 128 reading, our goal is to predict one of the six activities being performed by user.

### SCOPE:

Human activity recognition (HAR) can benefit various applications, such as health-care services and smart home applications. Many sensors have been utilized for human activity recognition, such as wearable sensors, smartphones, radio frequency (RF) sensors (WiFi, RFID), LED light sensors, cameras, etc. Owing to the rapid development of wireless sensor network, a large amount of data has been collected for the recognition of human activities with different kind of sensors. Conventional shallow learning algorithms, such as support vector machine and random forest, require to manually extract some representative features from large and noisy sensory data. 
Recently, deep learning has achieved great success in many challenging research areas, such as image recognition and natural language processing. The key merit of deep learning is to automatically learn representative features from massive data

### OBJECTIVE/AIM: 

Since there are many sensors in built in our smartphone to measure our position movements and orientation And because of these sensors the improvement in our daily life of human increases. Main objective of our project hey you should recognize the human’s activity by analysing the mobile phone’s sensor data.
More specifically, we have to make a model which can predict or accurately classifies whether a person is performing the action of laying, walking, walking upstairs, walking downstairs, setting or standing only on the basis of mobile phone sensor data.
This human activity recognition proposes many different applications and several benefits.
This mobile based application can be beneficial for all people. This application works as it tracks our activities overtime.
Our project falls into the scope of activity recognition, a field that offers many benefits and enables many new applications, for ex: step counters on our phones, as well as applications for people and personal health monitoring.

### ALGORITHM:

![image](https://user-images.githubusercontent.com/70462853/127215677-749b0cbc-92f3-44b4-9f60-5bb7d63d8047.png)

### System Design

Wearable Sensor. As wearable sensors can directly and efficiently capture body movements, they are the most commonly used for human activity recognition. These sensors can be freely integrated into smartphones, watches, bands, and even clothes.
 Accelerometer. An accelerometer is a device used to measure acceleration which is the rate of change of the velocity of an object. The measuring unit is meters per second squared (𝑚/𝑠 2 ) or Gforces (𝑔). The sampling frequency is usually in the range of tens to hundreds of Hz. For recognizing human activities, accelerometers can be mounted on various parts of a body, such as the waist [8], arm [170], ankle [11], wrist [63], et al. There are three axes in an often-used accelerometer. Therefore, a tri-variate time series would be achieved through an accelerometer. 
 
Gyroscope. A gyroscope is a device that measures orientation and angular velocity. The unit of angular velocity is measured in degrees per second (°/𝑠). The sampling rate is also from tens to hundreds of Hz. A gyroscope is usually integrated with an accelerometer and amounted on the same body parts. In addition, a gyroscope has three axes as well. 

Magnetometer. A magnetometer is another widely used wearable sensor for activity recognition, which is generally assembled with an accelerometer and a gyroscope into an inertial unit. It measures the change of a magnetic field at a particular location. The measurement units are Tesla (𝑇 ), and the sampling rate is from tens to hundreds of Hz. Likewise, a magnetometer has three axes.

 Electromyography (EMG). An EMG sensor is used to evaluate and record the electrical activity produced by skeletal muscles. Different from the above three kinds of sensors, EMG sensors require to be attached directly to human skin. As a result, it is less commonly used in conventional scenarios but more suitable for fine-grained motions such as hand [190] or arm [157] movements and facial expressions. The EMG provides a univariate time series of signal amplitudes.
 
 Electrocardiography (ECG). ECG is another biometric tool for activity recognition that measures the electrical activities generated by the heart. It also requires the sensor to contact the human’s skin directly. As different people’s hearts vibrate in significantly different ways, the ECG signals are difficult for processing subject variations. An ECG sensor provides a univariate time series data.
 
Feature Extraction

After collecting the data, it had to go through a transformation process in order to extract features that provide all the necessary information to the algorithm used for ML. For every set of readings, we computed five types of features, each generating a number of inputs for the learning algorithm. A brief description of the features can be found below.

Average

There were nine inputs for this feature, which represented the average value of readings per axis, computed as follows (where N is the number of readings for each sensor per 10 s, for this and all the following equations):

1N∑i=1Nxi

(1)
Average Absolute Difference

This feature (also with nine inputs) is the average absolute difference between the value of each of the readings and the mean value, for each axis, computed as:

1N∑i=1N|xi−μ|

(2)
Standard Deviation

The standard deviation was employed to quantify the variation of readings from the mean value, for each axis (resulting in nine inputs):

1N∑i=1Nxi−μ−−−−−−−−−−−⎷

(3)
Average Resultant Acceleration

This feature, having the inputs, was computed as the average of the square roots of the sum of the squared value of each reading:

1N∑i=1Nxi2+yi2+zi2−−−−−−−−−−−√

  (4)
Histogram

Finally, the histogram implies finding the marginal values for each axis (minimum -maximum), dividing that range into ten equal-sized intervals and determining what percentage of readings fall within each of the intervals (resulting in 90 inputs):

1N∑i=1N[(xiinbj)→1,j=1…10]

![image](https://user-images.githubusercontent.com/70462853/127683296-e904f453-dde8-4d6c-ad21-89773de270f8.png)


### Dataset link: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

