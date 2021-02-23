import os
import pickle
import tempfile

from google.cloud import storage as gcs


class GCSOperator():
    def __init__(self, project_id, bucket_name, credentials=None):
        """GCSOperator GSC wrapper class

        This class provides following APIs.

        - upload file
        - delete file
        - download file
        - load pickle
        - GCS file path

        Parameters
        ----------
        project_id : str
            GoogleCloudPlatform Project ID
        bucket_name : str
            GoogleCloudStorage Bucket Name
        credentials : str, optional
            GoogleCloudPlatform Credential Information, by default None
        """

        self._client = gcs.Client(project_id, credentials=credentials)

        self._bucket_name = bucket_name
        self._bucket = self._client.get_bucket(bucket_name)

        self._suffix = 'gs://'

    def upload_file(self,
                    gcs_path,
                    local_path,
                    bucket_name=None,
                    delete_local=False):
        """upload_file

        upload a file to GCS

        Parameters
        ----------
        gcs_path : str
            GCS file path
        local_path : str
            local file path
        bucket_name : str, optional
            GoogleCloudStorage Bucket Name
            if None, use default bucket, by default None
        delete_local : bool, optional
            delete local file option, by default False
        """

        if bucket_name is None:
            blob = self._bucket.blob(gcs_path)
        else:
            bucket = self._client.get_bucket(bucket_name)
            blob = bucket.blob(gcs_path)

        # upload file
        blob.upload_from_filename(local_path)
        print(f'Upload {local_path} to {gcs_path}')

        if delete_local:
            # remove local files
            os.remove(local_path)
            print(f'Delete {local_path}')

    def delete_file(self, gcs_path, bucket_name=None):
        """delete_file

        delete GCS file.

        Parameters
        ----------
        gcs_path : str
            GCS file path
        bucket_name : str, optional
            GoogleCloudStorage Bucket Name
            if None, use default bucket, by default None
        """

        if bucket_name is None:
            blob = self._bucket.blob(gcs_path)
        else:
            bucket = self._client.get_bucket(bucket_name)
            blob = bucket.blob(gcs_path)

        # delete files
        blob.delete()
        print(f'Delete {gcs_path} in the GCS')

    def download_file(self, gcs_path, local_path, bucket_name=None):
        """download_file

        download a GCS file to local.

        Parameters
        ----------
        gcs_path : str
            GCS file path
        local_path : str
            local file path
        bucket_name : str, optional
            GoogleCloudStorage Bucket Name
            if None, use default bucket, by default None
        """

        if bucket_name is None:
            blob = self._bucket.blob(gcs_path)
        else:
            bucket = self._client.get_bucket(bucket_name)
            blob = bucket.blob(gcs_path)

        blob.download_to_filename(local_path)

        print(f'Download {gcs_path} to {local_path}')

    def is_exist(self, gcs_path, bucket_name=None):

        if bucket_name is None:
            blob = self._bucket.blob(gcs_path)
        else:
            bucket = self._client.get_bucket(bucket_name)
            blob = bucket.blob(gcs_path)

        return blob.exists()

    def load_pickle(self, gcs_path, bucket_name=None):
        """load_pickle

        load pickle file without download.
        This method is intended for ML models.

        Parameters
        ----------
        gcs_path : str
            GCS file path
        bucket_name : str, optional
            GoogleCloudStorage Bucket Name
            if None, use default bucket, by default None

        Returns
        -------
        model : scikit-learn model
            trained ML model
        """

        if bucket_name is None:
            blob = self._bucket.blob(gcs_path)
        else:
            bucket = self._client.get_bucket(bucket_name)
            blob = bucket.blob(gcs_path)

        with tempfile.TemporaryFile() as fp:
            # download blob into temp file
            blob.download_to_file(fp)
            fp.seek(0)

            # load into joblib
            model = pickle.load(fp)
            print('load model')

        return model

    def get_fullpath(self, gcs_path, bucket_name=None):
        """get_fullpath get GCS full path

        Parameters
        ----------
        gcs_path : str
            GCS path, deeper than bucket_name
        bucket_name : str, optional
            GCS bucket name, by default None

        Returns
        -------
        full_path : str
            GCS full path
        """

        if bucket_name is None:
            full_path = os.path.join(self._suffix, self._bucket_name, gcs_path)

            return full_path
        else:
            full_path = os.path.join(self._suffix, bucket_name, gcs_path)

            return full_path

    def show_bucket_names(self):
        """show_bucket_names
        """
        [print(bucket.name) for bucket in self._client.list_buckets()]

    def show_file_names(self):
        """show_file_names
        """
        [print(file.name) for file in self._client.list_blobs(self._bucket)]
