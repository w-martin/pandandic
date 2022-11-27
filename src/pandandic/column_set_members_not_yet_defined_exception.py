class ColumnSetMembersNotYetDefinedException(Exception):
    def __init__(self, column_set_name: str):
        self._column_set_name = column_set_name

    def __str__(self) -> str:
        return f"Error. ColumnSet with name {self._column_set_name} was used with a member list that is not defined."
