#Packages load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder  # <<< UPDATED LINE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, mean_absolute_error,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import pickle
try:
    # Load the Dataset
    df = pd.read_csv("csv file/city_day.csv")
    df.head()
except FileNotFoundError as e:
    print(f"Error {e}")
except Exception as f:
    print(f"Error {f}")

print(f"Total Columns of this dataset is {df.shape[1]} and the Total Row is {df.shape[0]}.")

df.dtypes
print("\nLets Describe the Dataset Values.\n")
df.describe()

# Check and resolve null values
print("Check the Null values and resolve it.")
df.isnull().sum()

try:
    print("Lets Resolve the Missing Values.\n")
    for i in df.columns:
        if df[i].dtype == 'object':
            df[i] = df[i].fillna(df[i].mode()[0])
        else:
            df[i] = df[i].fillna(df[i].mean())

    df.isnull().sum()
except Exception as e:
    print(f"Error {e}")

# Change Date column into separate Day, Month, Year columns
try: 
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        df["Day"] = df["Date"].dt.dayofweek
except Exception as e:
    print(f"Error {e}")

# Properly drop Date column
df = df.drop(columns="Date", axis=1)

try:    
    # Reindex columns
    print("Lets rearrange the Order of the Column into this dataset.")
    new_order = ['City', 'Day', 'Month', 'Year', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
                'Benzene', 'Toluene', 'Xylene', 'AQI', 'AQI_Bucket']
    df = df.reindex(columns=new_order)
except Exception as e:
    print(f"Error {e}")

print("\nLets Check the Reindexed Datasets.\n")
df.head(10)

df['City'].unique()

df['AQI_Bucket'].unique()

try:
    # Separate numeric and object columns
    temp2 = [] #Object
    temp3 = [] #Numaric
    for i in df.columns:
        if df[i].dtype == 'object':
            temp3.append(i)
        else:
            temp2.append(i)

    print("Object Types od Columns:\n",temp3)
    print("Numaric Types od Columns:\n",temp2)

except Exception as e:
    print(f"Error {e}/")

try:
    # Check and visualize outliers
    print("Lets Check the Outliers of all Columns.\n")
    plt.figure(figsize=(15, 3 * len(temp2)))
    for j in range(1, len(temp2) + 1):
        plt.subplot(len(temp2), 1, j)
        sns.boxplot(x=temp2[j - 1], data=df)
        plt.title(f"{j}")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error {e}")

# Encode categorical columns
print("\nLets Encode the Categorical Data into Numeric format.\n")

#We are use the OrdinalEncoder function for the encode a City Columns into the Datasets.
city_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[['City']] = city_encoder.fit_transform(df[['City']])

try:
    # Encode target column AQI_Bucket
    lb_iqr = LabelEncoder()
    y = df["AQI_Bucket"]
    lb_iqr.fit(y) 
    y = lb_iqr.transform(y)
except Exception as e:
    print(f"Error {e}")

# Prepare features
x = df.drop(columns=["AQI_Bucket"])

try:
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
except Exception as e:
   print(f"Erro {e}")

try: 
    # Handle class imbalance using SMOTE
    sm = SMOTE(random_state=42)
    x_train, y_train = sm.fit_resample(x_train, y_train)
except Exception as e:
    print(f"Erro {e}")

try:
    # Scaling features for KNN
    std = StandardScaler()
    x_train_scaled = std.fit_transform(x_train)
    x_test_scaled = std.transform(x_test)
except Exception as e:
    print(f"Erro {e}")

print("\nLet see the Training Shape of the dataset.\n")
print(f"Total Training Data: {x_train_scaled.shape[0]} Rows\nTotal Testing Data: {x_test_scaled.shape[0]} Rows")

try:
    # Use KNN Model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train_scaled, y_train)
    y_predict_knn = knn.predict(x_test_scaled)

    print("\nThe Classification Report of this KNN Model is....\n", classification_report(y_test, y_predict_knn))
    print("\nThe Accuracy of the KNN Model is : ", round(accuracy_score(y_test, y_predict_knn) * 100, 2), "%")
    print("\nMean Absolute Error of this KNN Model:", round(mean_absolute_error(y_test, y_predict_knn), 2))
    print("\nMean Squared Error of this KNN Model :", round(mean_squared_error(y_test, y_predict_knn), 2))
    print("\nThe Confustion Matrix is:\n",confusion_matrix(y_test, y_predict_knn))
except Exception as e:
    print(f"Erro {e}")

try:
      # Use Decision Tree Classification Model
      dtree = DecisionTreeClassifier(max_depth=6, random_state=42)
      dtree.fit(x_train_scaled, y_train)
      y_tree_predict = dtree.predict(x_test_scaled)

      print("\nLet see the Classification Report of this Decision Tree Classifier Model...\n",
            classification_report(y_test, y_tree_predict))
      print("\n Accuracy of Decision Tree Classifier Model : ", round(accuracy_score(y_test, y_tree_predict) * 100, 2), "%")
      print("\nMean Absolute Error of this Decision Tree Classifier Model:", round(mean_absolute_error(y_test, y_tree_predict), 2))
      print("\nMean Squared Error of this Decision Tree Classifier Model :", round(mean_squared_error(y_test, y_tree_predict), 2))
      print("\nThe Confustion Matrix is:\n",confusion_matrix(y_test, y_tree_predict))
except Exception as e:
    print(f"Erro {e}")

df.head()

try:
    # User Input
    print("\nEnter All the Values of your Environment for the Correct Prediction.\n")

    def UserInput():
        City = input("Enter the City Name : ")
        Day = int(input("Enter the Date (Like 12, 15, 29) : "))
        Month = int(input("Enter the Month of Date (Like Jan=1, Feb=2...) : "))
        Year = int(input("Enter the Year of the Date (Like 2015, 2025) : "))
        PM25 = float(input("Enter the 'PM2.5' : "))
        PM10 = float(input("Enter the 'PM10' : "))
        NO = float(input("Enter the 'NO': "))
        NO2 = float(input("Enter the 'NO2': "))
        NOx = float(input("Enter the 'NOx': "))
        NH3 = float(input("Enter the 'NH3': "))
        CO = float(input("Enter the 'CO': "))
        SO2 = float(input("Enter the 'SO2': "))
        O3 = float(input("Enter the 'O3': "))
        Benzene = float(input("Enter the Benzene: "))
        Toluene = float(input("Enter the Toluene: "))
        Xylene = float(input("Enter the Xylene :"))
        AQI = float(input("Enter the AQI :"))

        user_input_data = {
            "City": City,
            "Day": Day,
            "Month": Month,
            "Year": Year,
            "PM2.5": PM25,
            "PM10": PM10,
            "NO": NO,
            "NO2": NO2,
            "NOx": NOx,
            "NH3": NH3,
            "CO": CO,
            "SO2": SO2,
            "O3": O3,
            "Benzene": Benzene,
            "Toluene": Toluene,
            "Xylene": Xylene,
            "AQI": AQI
        }
        
        user_df = pd.DataFrame([user_input_data])
        
        user_df['City'] = city_encoder.transform(user_df[['City']]) 

        # Encode & Scale user input
        user_df_scaled = std.transform(user_df)

        # Make Predictions
        knn_pred_num = knn.predict(user_df_scaled)[0]
        tree_pred_num = dtree.predict(user_df_scaled)[0]

        # Convert numeric predictions to original labels
        knn_pred_label = lb_iqr.inverse_transform([knn_pred_num])[0]
        dtree_pred_label = lb_iqr.inverse_transform([tree_pred_num])[0]

        print("\n==========Prediction Result==========\n")
        print(f"\nKNN Model predicts the Environment is: {knn_pred_label} ({knn_pred_num})")
        print(f"\nDecision Tree Model predicts the Environment is: {dtree_pred_label} ({tree_pred_num})")
        print("\n=====================================\n")


    UserInput()
except Exception as e:
    print(f"Erro {e}")

try:
    #Make the Pickle Model
    print("Lets Make the Pickle Model for the Fast Api.......\n")

    models = {"Ordinal_city_encoder":city_encoder,
            "label_encoder":lb_iqr,
            "standard_scaler":std,
            "knn_model":knn,
            "dtree_model":dtree   
    }
        
    with open('model/all_models.pkl','wb') as f:
        pickle.dump(models,f)
except Exception as e:
    print(f"Erro {e}")

try:
    #Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none',label='Correlation')
    plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
    plt.title("Correlation Heatmap", fontsize=15)
    plt.tight_layout()
    plt.savefig("Graph images/correlation_heatmap.png",dpi=400,bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Erro {e}")

try:
    #Model Accuracy Comparison

    model_names = ['KNN', 'Decision Tree']
    accuracies = [
        round(accuracy_score(y_test, y_predict_knn) * 100, 2),
        round(accuracy_score(y_test, y_tree_predict) * 100, 2)
    ]

    plt.figure(figsize=(6, 5))
    plt.bar(model_names, accuracies, color=['blue', 'green'])
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.5, f"{acc}%", ha='center')
    plt.tight_layout()
    plt.savefig("Graph images/model_accuracy_comparison.png",dpi=400,bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Erro {e}")

try:

    #Prediction Comparison

    labels = ["Correct Predictions", "Incorrect Predictions"]

    knn_correct = accuracy_score(y_test, y_predict_knn) * len(y_test)
    knn_incorrect = len(y_test) - knn_correct

    dtree_correct = accuracy_score(y_test, y_tree_predict) * len(y_test)
    dtree_incorrect = len(y_test) - dtree_correct

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, [knn_correct, knn_incorrect], width, label='KNN', color='skyblue')
    plt.bar(x + width/2, [dtree_correct, dtree_incorrect], width, label='Decision Tree', color='orange')

    plt.ylabel("Number of Predictions")
    plt.title("Prediction Comparison Between Models")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Graph images/prediction_comparison.png",dpi=400,bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Erro {e}")

try:
    # 📈 Line Plot for Training vs Testing Data
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]

    plt.figure(figsize=(7, 5))
    plt.plot(['Training Data', 'Testing Data'], [train_size, test_size], 
            marker='o', linestyle='-', color='purple', linewidth=2, markersize=8)

    plt.title("Training vs Testing Data Distribution", fontsize=14)
    plt.xlabel("Dataset Type", fontsize=12)
    plt.ylabel("Number of Records", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("Graph images/train_test_lineplot.png", dpi=400, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Erro {e}")

try:
    # 📈 Line Plot for Training vs Testing Data
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]

    plt.figure(figsize=(7, 5))
    plt.plot(['Training Data', 'Testing Data'], [train_size, test_size], 
            marker='o', linestyle='-', color='purple', linewidth=2, markersize=8)

    plt.title("Training vs Testing Data Distribution", fontsize=14)
    plt.xlabel("Dataset Type", fontsize=12)
    plt.ylabel("Number of Records", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("Graph images/train_test_lineplot.png", dpi=400, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Erro {e}")
