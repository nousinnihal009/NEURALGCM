import threading, sys, time, traceback, os

# Mock truststore block
class BlockTruststore:
    @classmethod
    def find_spec(cls, fullname, path, target=None):
        if 'truststore' in fullname:
            raise ImportError('Blocked')
        return None
sys.meta_path.insert(0, BlockTruststore())

def run_pip():
    import runpy, sys
    sys.argv = ['pip', 'install', 'certifi', '--no-cache-dir', '--disable-pip-version-check']
    try:
        runpy.run_module('pip', run_name='__main__')
    except Exception as e:
        print("Pip exited with:", e)

t = threading.Thread(target=run_pip)
t.daemon = True
t.start()
time.sleep(4)
print("\n--- DUMPING THREAD STACKS ---")
for th in threading.enumerate():
    if th is not threading.current_thread():
        print(f"Thread: {th.name} (ID: {th.ident})")
        traceback.print_stack(sys._current_frames()[th.ident])
print("--- END DUMP ---")
os._exit(1)
