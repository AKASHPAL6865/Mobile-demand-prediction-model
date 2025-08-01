from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

os.makedirs('static/plots', exist_ok=True)

df = pd.read_csv('mobile_demand.csv')

# ✅ Data Preprocessing
df['Demand_Label'] = ((df['Rating'] >= 4.0) &
                      (df['Storage'] >= 128) &
                      (df['advertisement_level'] >= 5) &
                      (df['Selling Price'] <= 200000)).astype(int)

# ✅ Define features and target
X = df[['battery_mah', 'advertisement_level', 'Storage', 'Memory', 'camera_mp', 'screen_size_inch', 'Rating']]
y = df['Demand_Label']

# ✅ Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# ✅ Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        def safe_int(value, default=0):
            try:
                return int(value)
            except:
                return default

        def safe_float(value, default=0.0):
            try:
                return float(value)
            except:
                return default

        input_data = {
            'battery_mah': safe_int(request.form.get('battery_mah')),
            'advertisement_level': safe_int(request.form.get('advertisement_level')),
            'Storage': safe_int(request.form.get('Storage')),
            'Memory': safe_int(request.form.get('Memory')),
            'camera_mp': safe_int(request.form.get('camera_mp')),
            'screen_size_inch': safe_float(request.form.get('screen_size_inch')),
            'Rating': safe_float(request.form.get('Rating')),
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df).max() * 100
        demand_result = "📱 In Demand" if prediction == 1 else "❌ Not in Demand"

        # ✅ Nearest Brand Matching using Euclidean Distance
        feature_cols = ['battery_mah', 'advertisement_level', 'Storage', 'Memory', 'camera_mp', 'screen_size_inch', 'Rating']
        distances = df[feature_cols].apply(lambda row: distance.euclidean(row, list(input_data.values())), axis=1)
        closest_index = distances.idxmin()
        predicted_brand = df.loc[closest_index, 'Brands']

        # ✅ Plot 1: Battery vs Storage
        grouped1 = df.groupby(['Storage', 'Demand_Label'])['battery_mah'].mean().reset_index()
        plt.figure(figsize=(5, 3))
        sns.lineplot(data=grouped1, x='Storage', y='battery_mah', hue='Demand_Label', marker='o')
        plt.scatter(input_data['Storage'], input_data['battery_mah'], color='black', s=100, label='Your Input')
        plt.title("Battery vs Storage by Demand")
        plt.legend()
        plt.tight_layout()
        plt.savefig('static/plots/plot1.png')
        plt.close()

        # ✅ Plot 2: Advertisement vs Rating
        plt.figure(figsize=(5, 3))
        sns.scatterplot(data=df, x='advertisement_level', y='Rating', hue='Demand_Label', palette='Set2')
        plt.scatter(input_data['advertisement_level'], input_data['Rating'], color='black', s=100, label='Your Input')
        plt.title("Ad Level vs Rating")
        plt.legend()
        plt.tight_layout()
        plt.savefig('static/plots/plot2.png')
        plt.close()

        # ✅ Plot 3: Price by Demand
        grouped2 = df.groupby('Demand_Label')['Selling Price'].mean().reset_index()
        plt.figure(figsize=(5, 3))
        sns.barplot(data=grouped2, x='Demand_Label', y='Selling Price', palette='coolwarm')
        plt.title("Average Price by Demand")
        plt.tight_layout()
        plt.savefig('static/plots/plot3.png')
        plt.close()

        # ✅ Plot 4: Brand-wise Demand Count
        plt.figure(figsize=(5, 3))
        ax = sns.countplot(data=df, x='Brands', hue='Demand_Label', palette='Set3')
        xpos = list(df['Brands'].unique()).index(predicted_brand)
        plt.axvline(x=xpos, color='black', linestyle='--', label='Predicted Brand')
        plt.title("Brand-wise Demand Count")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('static/plots/plot4.png')
        plt.close()

        # ✅ Plot 5: Demand Count with Your Marker
        plt.figure(figsize=(5, 3))
        sns.countplot(data=df, x='Demand_Label', palette='pastel')
        plt.title('Demand vs Not Demand Count')
        plt.xticks(ticks=[0, 1], labels=['Not in Demand', 'In Demand'])
        pred_label = prediction
        plt.scatter(pred_label, df['Demand_Label'].value_counts()[pred_label] + 10,
                    color='red', s=150, label='Your Input', marker='*')
        plt.legend()
        plt.tight_layout()
        plt.savefig('static/plots/plot5.png')
        plt.close()

        return render_template('result.html',
                               prediction_text=f"{demand_result}<br>🔢 Confidence: {confidence:.2f}%",
                               accuracy=accuracy * 100,
                               input_data=input_data,
                               brand=predicted_brand)

    except Exception as e:
        return f"Prediction failed: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
