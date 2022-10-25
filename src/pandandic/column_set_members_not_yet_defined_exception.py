from .column_set import ColumnSet


class ColumnSetMembersNotYetDefinedException(Exception):
    def __init__(self, column_set: ColumnSet):
        self._column_set = column_set

    def __str__(self) -> str:
        return f"Error. ColumnSet with name {self._column_set.name} was used with a member list that is not defined."
