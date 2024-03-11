import subprocess

# Function to run wily command
def run_wily(directory):
    try:
        # Build the wily cache (historical record of metrics)
        subprocess.run(['wily', 'build', directory], check=True)
        
        # Generate a report on the latest revision
        # The output can be captured in stdout for further processing
        result = subprocess.run(['wily', 'report', directory], capture_output=True, text=True, check=True)
        
        # Return the result of the wily report
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running Wily: {e}")
        return ""

# Directory containing the Python code to analyze
your_project_directory = 'main.py'

# Run wily and get the report
report = run_wily(your_project_directory)

# Check if the report contains data
if report:
    # Write the report to a file
    with open('wily_report.txt', 'w') as file:
        file.write(report)
    print("Wily report written to 'wily_report.txt'")
else:
    print("No report generated.")