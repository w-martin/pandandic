class DefinedLater:
    """
    Denotes that a Column alias or ColumnSet member list will be defined dynamically at runtime.
    If a read call is made before the DefinedLater is replaced, a ColumnAliasNotYetDefinedException or
    ColumnSetMembersNotYetDefinedException will be thrown.
    """
    ...
