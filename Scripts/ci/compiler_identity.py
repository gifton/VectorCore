"""Exact reviewed compiler pairs; CompilerIdentityTests exercise pair/alias rejection."""
import hashlib
from pathlib import Path

LOCAL_FRONTEND_SHA256 = '448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f'
LOCAL_DRIVER_SHA256 = '8635f454b0437014e3e03ccbaefb78c312d8d0337302f8a2348c15e59044762d'
HOSTED_FRONTEND_SHA256 = '020a521df9efc127b6817a2ed2f1560155108f0a0ead1ee1fa98147f5944eb4f'
HOSTED_DRIVER_SHA256 = '51e35b74f76fe8df9d2a203a9edddf4b10fbd2e94e3036137babacc390d9901e'
REVIEWED_PAIRS = frozenset({(LOCAL_FRONTEND_SHA256, LOCAL_DRIVER_SHA256),
                          (HOSTED_FRONTEND_SHA256, HOSTED_DRIVER_SHA256)})
LEGACY_FRONTEND_SHA256 = LOCAL_FRONTEND_SHA256


def validate_pair(frontend, driver):
    if type(frontend) is not str or type(driver) is not str or (frontend, driver) not in REVIEWED_PAIRS:
        raise ValueError('unreviewed frontend/driver pair')


def validate_legacy_frontend(frontend):
    if type(frontend) is not str or frontend != LEGACY_FRONTEND_SHA256:
        raise ValueError('unreviewed legacy frontend; paired identity required')


def observe_tools(swiftc):
    swiftc = Path(swiftc)
    if swiftc.name != 'swiftc':
        raise ValueError('compiler entry point must be named swiftc')
    paths = {'frontendSHA256': swiftc.with_name('swift-frontend'),
             'swiftDriverSHA256': swiftc.with_name('swift-driver'),
             'swiftcSHA256': swiftc, 'swiftSHA256': swiftc.with_name('swift')}
    observed = {}
    for name, path in paths.items():
        with path.open('rb') as stream:
            observed[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return observed


def validate_tools(observed):
    expected = {'frontendSHA256', 'swiftDriverSHA256', 'swiftcSHA256', 'swiftSHA256'}
    if type(observed) is not dict or set(observed) != expected:
        raise ValueError('compiler file inventory differs')
    validate_pair(observed['frontendSHA256'], observed['swiftDriverSHA256'])
    if not (observed['swiftcSHA256'] == observed['swiftSHA256'] == observed['frontendSHA256']):
        raise ValueError('Swift entry-point fingerprint differs from frontend')
