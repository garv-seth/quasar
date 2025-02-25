class PromptTemplate:
    """A template for formatting prompts with variables."""
    
    def __init__(self, template: str):
        """
        Initialize the prompt template.
        
        Args:
            template (str): Template string with placeholders in {variable} format
        """
        self.template = template
        
    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables.
        
        Args:
            **kwargs: Key-value pairs for template variables
            
        Returns:
            str: Formatted prompt
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Missing required template variable: {e}")
        
    def __str__(self) -> str:
        return f"PromptTemplate(template='{self.template}')"
