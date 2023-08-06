
class Config(object):
    DEBUG = True
    TESTING = False

class DevelopmentConfig(Config):
    organization_KEY = "org-evaR5kuPlH2xUy0vHjaZG2MY"
    OPENAI_KEY = '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'
    

config = {
    'development': DevelopmentConfig,
    'testing': DevelopmentConfig,
    'production': DevelopmentConfig
}
