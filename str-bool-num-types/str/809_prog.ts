export interface Leaf<TLeaf> {
    kind: 'leaf';
    value: TLeaf;
}

export interface Node<TNode, TLeaf> {
    kind: 'node';
    childrens: (Node<TNode, TLeaf> | Leaf<TLeaf>)[];
    name?: string;
    value?: TNode;
}

export type Tree<TNode, TLeaf> = Node<TNode, TLeaf>;

export interface IntermediateNode<TNode> {
    name: <FILL>;
    value: TNode;
}

export const createTree = <TNode, TLeaf>(): Tree<TNode, TLeaf> => ({
    kind: 'node',
    childrens: []
});

export const findNode = <TNode, TLeaf>(
    tree: Tree<TNode, TLeaf>,
    path: readonly string[]
): Node<TNode, TLeaf> | undefined => {
    const remainingPath = [...path];
    remainingPath.reverse();

    let node: string | undefined;
    let previousNode = tree;
    while ((node = remainingPath.pop())) {
        const nextNode = previousNode.childrens
            .filter((child): child is Node<TNode, TLeaf> => child.kind === 'node')
            .find((child: Node<TNode, TLeaf>) => child.name === node);
        if (!nextNode) {
            return undefined;
        }

        previousNode = nextNode;
    }

    return previousNode;
};

export const insertNodes = <TNode, TLeaf>(
    tree: Tree<TNode, TLeaf>,
    nodes: readonly IntermediateNode<TNode>[]
): void => {
    const remainingNodes = [...nodes];
    remainingNodes.reverse();

    let previousNode = tree;
    while (remainingNodes.length > 0) {
        const node = remainingNodes.pop();
        if (!node) {
            return undefined;
        }

        let nextNode = previousNode.childrens
            .filter((child): child is Node<TNode, TLeaf> => child.kind === 'node')
            .find((child: Node<TNode, TLeaf>) => child.name === node.name);
        if (!nextNode) {
            nextNode = {
                kind: 'node',
                name: node.name,
                value: node.value,
                childrens: []
            };
            previousNode.childrens.push(nextNode);
        }

        previousNode = nextNode;
    }
};

export const insertLeaf = <TNode, TLeaf>(
    tree: Tree<TNode, TLeaf>,
    leaf: TLeaf,
    nodes: readonly IntermediateNode<TNode>[]
): void => {
    insertNodes(tree, nodes);

    const lastParent = findNode(
        tree,
        nodes.map((node) => node.name)
    );
    if (!lastParent) {
        throw new Error(`Failed to find parent node for leaf: ${leaf}`);
    }

    lastParent.childrens.push({
        kind: 'leaf',
        value: leaf
    });
};
