import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('sms_spam/SMSSpamCollection', sep='\t')
    print(df)
    print("Let's detect some spam")
