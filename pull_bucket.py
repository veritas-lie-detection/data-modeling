import pickle
import boto3
import boto3.session
import pandas as pd

cred = boto3.Session().get_credentials()
ACCESS_KEY = cred.access_key
SECRET_KEY = cred.secret_key
SESSION_TOKEN = cred.token  # optional

s3client = boto3.client('s3',
                        aws_access_key_id="AKIAWHU4CHOWC7RRQAFO",
                        aws_secret_access_key="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"
                        )


response = s3client.get_object(
    Bucket='veritas-lie-detection-dataset', Key='fraudulent/1143155/2009.pkl')


body = response['Body'].read()
data = pickle.loads(body)
# print(data)


def build_database(s3_client):
    """
    This function will list down all files in a folder from S3 bucket
    :return: None
    """
    df = pd.DataFrame()
    bucket_name = "veritas-lie-detection-dataset"
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    files = response.get("Contents")

    cnt = 0
    for file in files:
        response = s3client.get_object(
            Bucket='veritas-lie-detection-dataset', Key=str(file['Key']))
        body = response['Body'].read()
        data = pickle.loads(body)
        data['classification'] = str(file['Key'].split("/")[0])
        df_dictionary = pd.DataFrame([data])
        df = pd.concat([df, df_dictionary], ignore_index=True)

    return df
        # print(f"file_name: {file&#91;'Key']}, size: {file&#91;'Size']}")

df = build_database(s3client)
df.to_csv('df.csv')


