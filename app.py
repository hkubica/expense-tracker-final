from flask import Flask, render_template, redirect, url_for, request, flash, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import csv
import io
import os
import numpy as np
from sklearn.linear_model import LinearRegression


app = Flask(__name__)

#Path to file
db_path = os.path.join(os.getcwd(), 'data', 'db', 'expensescopy.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SECRET_KEY'] = os.urandom(24)  #Secure Secret Key made
#Ensure the data/db directory exists
os.makedirs(os.path.join(os.getcwd(), 'data', 'db'), exist_ok=True)


#Database and Login manager set up
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

#User model for authentication
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    monthly_goal = db.Column(db.Float, default=0)  # Column to store monthly spending goal

#Expenses Model Created
class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date)  
    category = db.Column(db.String(150))
    amount = db.Column(db.Float)
    description = db.Column(db.String(150))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

#Load user for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))  

#Route for the main page of website
@app.route('/')
@login_required
def index():
    #Retrieve the month and year currently
    current_date = datetime.now()
    current_month = current_date.month
    current_year = current_date.year

    #Retrieve the users expenses for the current month
    user_expenses = Expense.query.filter(
        Expense.user_id == current_user.id,
        db.extract('month', Expense.date) == current_month,
        db.extract('year', Expense.date) == current_year
    ).all()
    
    #Calculate the total spent this month
    total_spent = sum(expense.amount for expense in user_expenses)
    
    #Calculate the progress towards the monthly goal
    if current_user.monthly_goal > 0:
        progress_percentage = min((total_spent / current_user.monthly_goal) * 100, 100)
    else:
        progress_percentage = 0
    
    #Pass the calculated values to the template
    return render_template('index.html', progress_percentage=progress_percentage, total_spent=total_spent)

#Route for registration of user
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        #Check if the username has already been created and if it exists in the database
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))

        #Hash the password for security purposes
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        user = User(username=username, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

#Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials', 'error')
    return render_template('login.html')

#Route for user logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

#Route for adding expenses
@app.route('/add', methods=['POST'])
@login_required
def add_expense():
    if request.method == 'POST':
        date = request.form['date']
        category = request.form['category']
        amount = request.form['amount']
        description = request.form['description']

        #Convert date string to a date object
        date_obj = datetime.strptime(date, '%Y-%m-%d').date()

        new_expense = Expense(date=date_obj, category=category, amount=float(amount), description=description, user_id=current_user.id)
        db.session.add(new_expense)
        db.session.commit()
        return redirect(url_for('index'))

#Route for setting the monthly goal
@app.route('/set_goal', methods=['POST'])
@login_required
def set_goal():
    #Get the goal from the form
    monthly_goal = request.form.get('monthly_goal')
    if monthly_goal is not None and monthly_goal.strip() != "":
        try:
            current_user.monthly_goal = float(monthly_goal)
            db.session.commit()
            flash('Monthly spending goal updated successfully!', 'success')
        except ValueError:
            flash('Invalid value for monthly goal. Please enter a valid number.', 'error')
    else:
        flash('Please provide a valid monthly goal.', 'error')

    #Redirect to the main page with updated information
    return redirect(url_for('index'))

#Route for viewing expenses
@app.route('/expenses', methods=['GET', 'POST'])
@login_required
def view_expenses():
    query = Expense.query.filter_by(user_id=current_user.id)

    #Apply filtering based on form data
    if request.method == 'POST':
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        category = request.form.get('category')

        if start_date:
            query = query.filter(Expense.date >= start_date)
        if end_date:
            query = query.filter(Expense.date <= end_date)
        if category:
            query = query.filter_by(category=category)

    expenses = query.all()

    #Turn the list of Expense objects into something more readable: dictionaries
    expenses_list = [
        {
            'id': expense.id,
            'date': expense.date,
            'category': expense.category,
            'amount': expense.amount,
            'description': expense.description,
        } for expense in expenses
    ]

    #Pass the cleaned-up list to the template for rendering
    return render_template('expenses.html', expenses=expenses_list)

#Route to predict a budget based on user's entire history
@app.route('/predict_budget', methods=['GET'])
@login_required
def predict_budget():
    #Grab all expenses for the user that has logged in
    user_expenses = Expense.query.filter_by(user_id=current_user.id).all()

    #Make sure there is enough data to work with
    if len(user_expenses) < 2:
        return jsonify({'predicted_budget': 'Not enough data to make a prediction'})

    #Set up the data for training the model
    dates = list(range(len(user_expenses)))
    amounts = [expense.amount for expense in user_expenses]

    # Convert data to numpy arrays
    X = np.array(dates).reshape(-1, 1)  #Features e.g. (0, 1, 2, ...)
    y = np.array(amounts)  #Target values - amounts

    #Train a simple linear regression model on the user's data
    model = LinearRegression()
    model.fit(X, y)

    #Predict the budget for the next month
    next_month_index = np.array([[len(user_expenses)]])
    predicted_budget = model.predict(next_month_index)[0]

    #Sendthe predicted budget as a JSON response
    return jsonify({'predicted_budget': predicted_budget})

#Route in order to evaluate the model's performance
#@app.route('/evaluate', methods=['GET'])
#@login_required
#def evaluate_model():
    #Get all expenses for the evaluation
   # user_expenses = Expense.query.filter_by(user_id=current_user.id).all()

    #Make sure there is enougn data points for evaluation
    #if len(user_expenses) < 2:
        #return jsonify({'error': 'Not enough data to evaluate the model'})

    #Prepare the data ready for the evalutation
    #dates = list(range(len(user_expenses))) #Indices used as pseudo dates
    #amounts = [expense.amount for expense in user_expenses]

    #Convert the data to numpy arrays in order to train and test
    #X = np.array(dates).reshape(-1, 1)
    #y = np.array(amounts)

    #Train the model
    #model = LinearRegression()
    #model.fit(X, y)

    #Get predictions and calculate error metrics
    #predictions = model.predict(X)
    #mae = np.mean(np.abs(predictions - y))
    #rmse = np.sqrt(np.mean((predictions - y) ** 2))

    #Send back results as JSON
    #return jsonify({
        #'Mean Absolute Error (MAE)': round(mae, 2),
        #'Root Mean Squared Error (RMSE)': round(rmse, 2)
    #})
if __name__ == '__main__':
    import os
    with app.app_context():
        db.create_all()  #Ensure database tables are created
    #Use port 8080, required by OpenShift
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)  #Debug=False for production




