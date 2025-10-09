"""
Template Loader Module
Handles loading and rendering HTML templates for the Streamlit application.
"""

from pathlib import Path
from typing import Dict, Any


class TemplateLoader:
    """Loads and renders HTML templates from the templates directory."""
    
    def __init__(self, templates_dir: Path = None):
        """
        Initialize the template loader.
        
        Args:
            templates_dir: Path to the templates directory. 
                          If None, uses the directory of this file.
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent
        self.templates_dir = templates_dir
        self._cache = {}
    
    def load_template(self, template_name: str) -> str:
        """
        Load a template file from the templates directory.
        
        Args:
            template_name: Name of the template file (without .html extension)
        
        Returns:
            Template content as string
        """
        # Check cache first
        if template_name in self._cache:
            return self._cache[template_name]
        
        # Load from file
        template_path = self.templates_dir / f"{template_name}.html"
        
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template '{template_name}' not found at {template_path}"
            )
        
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Cache the template
        self._cache[template_name] = content
        
        return content
    
    def render_template(self, template_name: str, **kwargs: Any) -> str:
        """
        Load and render a template with the provided variables.
        
        Args:
            template_name: Name of the template file (without .html extension)
            **kwargs: Variables to substitute in the template
        
        Returns:
            Rendered template as string
        """
        template = self.load_template(template_name)
        return template.format(**kwargs)
    
    def clear_cache(self):
        """Clear the template cache."""
        self._cache.clear()


# Global template loader instance
_template_loader = TemplateLoader()


def get_template_loader() -> TemplateLoader:
    """Get the global template loader instance."""
    return _template_loader


def render_template(template_name: str, **kwargs: Any) -> str:
    """
    Convenience function to render a template using the global loader.
    
    Args:
        template_name: Name of the template file (without .html extension)
        **kwargs: Variables to substitute in the template
    
    Returns:
        Rendered template as string
    """
    return _template_loader.render_template(template_name, **kwargs)


def load_template(template_name: str) -> str:
    """
    Convenience function to load a template using the global loader.
    
    Args:
        template_name: Name of the template file (without .html extension)
    
    Returns:
        Template content as string
    """
    return _template_loader.load_template(template_name)

