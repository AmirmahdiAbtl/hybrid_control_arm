from src.web.routes.jobs import bp as jobs_bp
from src.web.routes.pages import bp as pages_bp


def register_routes(app):
    app.register_blueprint(pages_bp)
    app.register_blueprint(jobs_bp)
