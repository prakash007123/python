from flask import Flask, render_template, request, redirect, send_from_directory, url_for, session
from flask import Response
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import traceback
import mysql.connector
import os
import logging

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

log_format = '%(asctime)s [%(levelname)s] [%(module)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


# Log to a file in addition to the console
log_file = 'app.log'
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter(log_format))
app.logger.addHandler(file_handler)

# Dictionary to store results with unique identifiers
results_dict = {}
# Counter for generating unique identifiers
result_id_counter = 1

# MySQL database connection
db_connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Ch@ngep0nd@123',
    database='outlier'
)

# Simple in-memory user database (username: password) for demonstration purposes
# In a real-world app, the user information would be stored in the MySQL database
users = {
    'john': 'pass123',
    'jane': 'pass456'
}


# @app.route('/', methods=['GET', 'POST'])
# def home():
#     global result_id_counter
#     if 'username' not in session:
#         return redirect(url_for('login'))

#     if request.method == 'POST':
#         try:
#             # Get the uploaded dataset file
#             file = request.files['dataset']
#             # Read the dataset from the file
#             data = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))

#             # Fetch all columns except the last one (assuming the last column is the target variable)
#             columns_to_drop = data.columns[:-1]

#             # Drop unwanted columns
#             data = data.drop(columns_to_drop, axis=1)
#             print(f"Dropped columns: {columns_to_drop.tolist()}")

#             # Convert categorical variables into numeric representations
#             data = pd.get_dummies(data)

#             # Normalize the numeric variables
#             scaler = StandardScaler()
#             data_scaled = scaler.fit_transform(data)

#             # Apply PCA on the normalized data
#             pca = PCA(n_components=2)
#             reduced_data = pca.fit_transform(data_scaled)

#             # Fit One-Class SVM model
#             svm = OneClassSVM()
#             svm.fit(reduced_data)

#             # Get the threshold value from the form
#             threshold = float(request.form['threshold'])

#             # Calculate SVM scores
#             svm_scores = svm.decision_function(reduced_data)

#             # Find outliers based on the threshold
#             svm_outliers = np.where(svm_scores < threshold)[0]

#             # Plot the data points
#             plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color='blue', label='Data')
#             # Plot the outliers in a different color
#             plt.scatter(reduced_data[svm_outliers, 0], reduced_data[svm_outliers, 1], color='red', label='Outliers')
#             plt.legend()
            
#             # Generate a unique identifier for this result
#             result_id = f"result_{result_id_counter}"
#             result_id_counter += 1

#             # Store the result in the results_dict
#             results_dict[result_id] = {
#                 'svm_outliers': svm_outliers,
#                 'plot_file': f'static/plot_{result_id}.png',
#                 'outliers_info': []  # Initialize the list of outliers_info
#             }

#             # Get the outliers information as a list of dictionaries
#             for idx in svm_outliers:
#                 outlier_info = {
#                     'outlier_index': int(idx),
#                     'data_point': reduced_data[int(idx)].tolist()
#                 }
#                 results_dict[result_id]['outliers_info'].append(outlier_info)

#             plt.savefig(f'static/plot_{result_id}.png')
#             plt.close()

#             return render_template('result.html', outliers_info=results_dict[result_id]['outliers_info'], result_id=result_id)

#         except Exception as e:
#             # Log the error traceback
#             traceback.print_exc()
#             return "An error occurred while processing the data."

#     return render_template('index.html')
@app.route('/', methods=['GET', 'POST'])
def home():
    global result_id_counter
    if 'username' not in session:
        app.logger.info("User is not logged in. Redirecting to login page.")
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Get the uploaded dataset file
            file = request.files['dataset']
            
            # Log the file upload
            app.logger.info("File uploaded: %s", file.filename)
            
            # Read the dataset from the file
            data = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))

            # Fetch all columns except the last one (assuming the last column is the target variable)
            columns_to_drop = data.columns[:-1]

            # Drop unwanted columns
            data = data.drop(columns_to_drop, axis=1)
            app.logger.info("Dropped columns: %s", columns_to_drop.tolist())

            # Convert categorical variables into numeric representations
            data = pd.get_dummies(data)

            # Normalize the numeric variables
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # Apply PCA on the normalized data
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(data_scaled)

            # Fit One-Class SVM model
            svm = OneClassSVM()
            svm.fit(reduced_data)

            # Get the threshold value from the form
            threshold = float(request.form['threshold'])
            
            # Log the threshold value
            app.logger.info("Threshold value: %s", threshold)

            # Calculate SVM scores
            svm_scores = svm.decision_function(reduced_data)

            # Find outliers based on the threshold
            svm_outliers = np.where(svm_scores < threshold)[0]
            

            # Plot the data points
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color='blue', label='Data')
            # Plot the outliers in a different color
            plt.scatter(reduced_data[svm_outliers, 0], reduced_data[svm_outliers, 1], color='red', label='Outliers')
            plt.legend()
            
            # Generate a unique identifier for this result
            result_id = f"result_{result_id_counter}"
            result_id_counter += 1
            
            # Log the result ID
            app.logger.info("Result ID generated: %s", result_id)

            # Store the result in the results_dict
            results_dict[result_id] = {
                'svm_outliers': svm_outliers,
                'plot_file': f'static/plot_{result_id}.png',
                'outliers_info': []  # Initialize the list of outliers_info
            }

            # Get the outliers information as a list of dictionaries
            for idx in svm_outliers:
                outlier_info = {
                    'outlier_index': int(idx),
                    'data_point': reduced_data[int(idx)].tolist()
                }
                results_dict[result_id]['outliers_info'].append(outlier_info)

            plt.savefig(f'static/plot_{result_id}.png')
            plt.close()

            return render_template('result.html', outliers_info=results_dict[result_id]['outliers_info'], result_id=result_id)

        except Exception as e:
            # Log the error traceback
            app.logger.error("An error occurred while processing the data: %s", str(e), exc_info=True)
            return "An error occurred while processing the data."

    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')


    
@app.route('/clear_outliers/<result_id>', methods=['POST'])
def clear_outliers(result_id):
    if result_id in results_dict:
        if request.form['action'] == 'clear':
            selected_outliers = request.form.getlist('selected_outliers')
            selected_outliers = [int(idx) for idx in selected_outliers]
            # Filter out the selected outliers
            results_dict[result_id]['outliers_info'] = [outlier for outlier in results_dict[result_id]['outliers_info'] if outlier['outlier_index'] not in selected_outliers]

            # Update the plot based on the cleared data
            updated_svm_outliers = [outlier['outlier_index'] for outlier in results_dict[result_id]['outliers_info']]
            updated_plot_image_path = f'static/plot_{result_id}_updated.png'
            # Replot using the updated_svm_outliers
            # ...

            # Return the updated result page
            return render_template('result.html', outliers_info=results_dict[result_id]['outliers_info'], result_id=result_id, plot_image_path=updated_plot_image_path)

        elif request.form['action'] == 'download':
            # Prepare and send the cleared outliers dataset for download
            result_data = results_dict[result_id]
            cleared_outliers_df = pd.DataFrame([outlier['data_point'] for outlier in result_data['outliers_info']])
            cleared_outliers_csv = cleared_outliers_df.to_csv(index=False)
            response = Response(cleared_outliers_csv, content_type='text/csv')
            response.headers["Content-Disposition"] = f"attachment; filename=cleared_outliers_{result_id}.csv"
            return response

    return "Invalid result ID."




@app.route('/result/<result_id>')
def show_result(result_id):
    if 'username' not in session:
        return redirect(url_for('login'))

    if result_id in results_dict:
        # Fetch the outliers for the specified result_id
        svm_outliers = results_dict[result_id]['svm_outliers']

        # Construct the path to the plot image based on the result_id
        plot_image_path = f'static/plot_{result_id}.png'

        return render_template('result.html', outliers_info=results_dict[result_id]['outliers_info'], result_id=result_id, plot_image_path=plot_image_path)
    else:
        return "Invalid result ID."


@app.route('/download/<result_id>')
def download_result(result_id):
    if result_id in results_dict:
        # Fetch the result data for the specified result_id
        result_data = results_dict[result_id]

        # Create a CSV file containing the result data
        csv_filename = f'result_{result_id}.csv'
        csv_content = "Outlier Index,Data Point\n"
        for outlier_info in result_data['outliers_info']:
            outlier_index = outlier_info['outlier_index']
            data_point = ','.join(map(str, outlier_info['data_point']))
            csv_content += f"{outlier_index},{data_point}\n"

        # Create a Flask Response with the CSV content
        response = Response(csv_content, content_type='text/csv')
        response.headers["Content-Disposition"] = f"attachment; filename={csv_filename}"
        return response
    else:
        return "Invalid result ID."


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the user exists in the MySQL database
        cursor = db_connection.cursor()
        cursor.execute('SELECT * FROM users WHERE username=%s AND password=%s', (username, password))
        user_data = cursor.fetchone()
        cursor.close()

        if user_data:
            # Store the username in the session
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', message="Invalid username or password.")

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']  # Get the email value
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if the username already exists in the MySQL database
        cursor = db_connection.cursor()
        cursor.execute('SELECT * FROM users WHERE username=%s', (username,))
        existing_user = cursor.fetchone()
        cursor.close()

        if existing_user:
            return render_template('register.html', message="Username already exists. Please choose a different one.")
        elif password != confirm_password:
            return render_template('register.html', message="Passwords do not match. Please try again.")
        else:
            # Add the new user to the MySQL database
            cursor = db_connection.cursor()
            cursor.execute('INSERT INTO users (username, email, password) VALUES (%s, %s, %s)', (username, email, password))
            db_connection.commit()
            cursor.close()

            # Store the username in the session
            session['username'] = username
            return redirect(url_for('home'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    # Clear the session data
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    # Create the 'static' folder if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
