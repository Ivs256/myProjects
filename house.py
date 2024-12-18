from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        Square_Feet = float(request.form['Square_Feet'])
        Bedrooms = int(request.form['Bedrooms'])
        Floors = int(request.form['Floors'])
        Year = int(request.form['Year'])
        Distance_to_Center = float(request.form['Distance_to_Center'])
        processed_data =[Square_Feet,Bedrooms,Floors,Year,Distance_to_Center]
        prediction = model.predict([processed_data])
        result = f"Predicted price: {prediction[0]:,.2f}"
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
