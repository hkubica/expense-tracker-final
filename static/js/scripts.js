document.addEventListener("DOMContentLoaded", function() {
    // Fetch the expense data from the backend
    const expenseData = JSON.parse('{{ expenses | tojson | safe }}');
    
    const categories = {};

    // Categorize the expenses and sum the amounts per category
    expenseData.forEach(expense => {
        const category = expense[2]; // Assuming category is the third element in the expense array
        const amount = parseFloat(expense[3]); // Assuming amount is the fourth element in the expense array
        if (!isNaN(amount)) {
            categories[category] = (categories[category] || 0) + amount;
        }
    });
    
    // Prepare the data for the chart
    const chartData = {
        labels: Object.keys(categories),
        datasets: [{
            label: 'Spending by Category',
            data: Object.values(categories),
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
        }]
    };

    // Get the context of the canvas element and create the chart
    const ctx = document.getElementById('expenseChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: chartData,
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
});
