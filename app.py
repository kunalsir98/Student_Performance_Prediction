from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    else:
        try:
             # Collect data from the form
            data = CustomData(
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score')),
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course')
            )

            # Convert data into DataFrame
            final_new_data = data.get_data_as_dataframe()

            # Initialize the prediction pipeline
            predict_pipeline = PredictPipeline()

            # Make prediction
            prediction = predict_pipeline.predict(final_new_data)

            # Extract the result
            predicted_result = prediction[0]

            # Render the result template
            return render_template('results.html', final_result=predicted_result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
