import time


def fibonacci_timed():
    # Set time limit to 5 minutes (300 seconds)
    end_time = time.time() + 300

    # Initialize first two numbers
    a, b = 0, 1
    count = 0

    try:
        while time.time() < end_time:
            print(f"Fibonacci number {count}: {a}")
            # Calculate next number
            a, b = b, a + b
            count += 1

            # Optional: Add a small delay to prevent overwhelming output
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nProgram stopped by user")

    print(f"\nReached {count} Fibonacci numbers in {int(time.time() - (end_time - 300))} seconds")


if __name__ == "__main__":
    print("Starting Fibonacci sequence for 5 minutes...")
    fibonacci_timed()
