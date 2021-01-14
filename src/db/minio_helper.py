from minio import Minio
import os
import logbook
from utils.log_helper import g_log_helper
g_log = g_log_helper.make_logger(logbook.INFO)

SERVER_ADDR   = os.environ.get('MINIO_SERVER_ADDR', '192.168.135.32:9000')
ACCESS_KEY    = os.environ.get('MINIO_ACCESS_KEY', 'minioadmin')
SECRET_KEY    = os.environ.get('MINIO_SECRET_KEY', 'minioadmin')
HTTP_PROTOCOL = os.environ.get('MINIO_HTTP_PROTOCOL', 'http')


class MinioHelper(object):
    def __init__(self, access_key=ACCESS_KEY, secret_key=SECRET_KEY, server_addr=SERVER_ADDR):
        self.server_addr = server_addr
        g_log.info('server_addr: %s' % SERVER_ADDR)
        self.minio_client = Minio(server_addr,
                                  access_key=access_key,
                                  secret_key=secret_key,
                                  secure=False)

    def upload_file(self, file_path, object_name, bucket_name='panos', content_type=''):
        ret_dict = {
            'code': 0,
            'file_url': '',
            'code_msg': '',
        }
        try:
            etag = self.minio_client.fput_object(bucket_name=bucket_name,
                                                 file_path=file_path,
                                                 object_name=object_name,
                                                 content_type=content_type)
            ret_dict['etag'] = etag
            ret_dict['public_url'] = HTTP_PROTOCOL + '://' + self.server_addr + '/' + bucket_name + '/' + object_name
            ret_dict['presigned_url'] = self.minio_client.presigned_get_object(bucket_name=bucket_name,
                                                                               object_name=object_name)
            return ret_dict
        except Exception as e:
            ret_dict['code'] = 1
            ret_dict['code_msg'] = str(e)
            return ret_dict

    def file_exist(self, object_name, bucket_name='panos'):
        try:
            data = self.minio_client.get_object(bucket_name, object_name)
            return True
        except Exception as err:
            print(err)
            return False


if __name__ == '__main__':
    minio_helper = MinioHelper()
    ret_dict = minio_helper.upload_file(file_path='./09005700121902131735110055A.jpg', object_name='09005700121902131735110055A.jpg')
    ret_dict = minio_helper.upload_file(file_path='config.yaml', object_name='config.yaml')
    print(ret_dict)

    