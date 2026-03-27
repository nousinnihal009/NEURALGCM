import sys
class BlockTruststore:
    @classmethod
    def find_spec(cls, fullname, path, target=None):
        if 'truststore' in fullname:
            raise ImportError('Blocked to prevent Windows CryptoAPI deadlock')
        return None

sys.meta_path.insert(0, BlockTruststore())
import runpy
sys.argv = ['pip'] + sys.argv[1:]
runpy.run_module('pip', run_name='__main__')
