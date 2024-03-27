/*

*/

#[macro_use]
extern crate pest_derive;
#[allow(dead_code)]
use pest::Parser;
use std::fs;
use std::io::{self,};
use std::path::{Path, PathBuf};
use std::env;
use std::fs::File;
use rand::seq::SliceRandom;
extern crate regex;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;
use csv::Writer;
use std::error::Error;
#[derive(Clone, Debug)]


struct Node {
    name: String,
    left_child: Option<Box<Node>>,
    right_child: Option<Box<Node>>,
    parent: Option<usize>, // Using index for parent
    depth: Option<f64>,
    length: f64,
}
#[derive(Clone, Debug)]
struct FlatNode {
    name: String,
    left_child: Option<usize>,
    right_child: Option<usize>,
    parent: Option<usize>,
    depth: Option<f64>,
    length: f64,
}
#[derive(Parser)]
#[grammar = "newick.pest"]
struct NewickParser;

// --------------------------------

fn give_depth(node: &mut Node, depth: f64) {
    /*
    Recursively gives the depth of nodes in a "Node" arborescent tree.
    ----------------------------------
    Input:
    - A node, and its depth in the tree
    ----------------------------------
    Output:
    - None. The tree is modified in place via a mutable reference (&mut)
    */
    node.depth = Some(depth);
    if let Some(left_child) = &mut node.left_child {
        give_depth(left_child, depth + left_child.length);
    }
    if let Some(right_child) = &mut node.right_child {
        give_depth(right_child, depth + right_child.length);
    }
}
// The following two functions are used to convert a newick string into an arborescent Tree structure, where each "Node" object owns its two children in a Box object.
fn newick_to_tree(pair: pest::iterators::Pair<Rule>) -> Vec<Node> {
    /*
    The input .nwk file of the script is a concatenation of newick trees, separated by ";".
    This function returns a vector containing each of these trees.
    --------------------------------
    Input:
    - A "pair" object of type "newick" (see newick.pest grammar) which is a newick string representing one single tree.
    ----------------------------------
    Output:

    */
    let mut vec_trees:Vec<Node> = Vec::new();
    for inner in pair.into_inner() {
        let tree = handle_pair(inner);
        vec_trees.push(tree.unwrap());
    }
    return vec_trees
}
fn handle_pair(pair: pest::iterators::Pair<Rule>) -> Option<Node> {
    /* 
    Recursively parses a newick string representing a single newick tree, according to the grammar contained in newick.pest.
    ----------------------------------
    Input:
    - A "pair" object representing a part of this string.
    ----------------------------------
    Output:
    - Either a node if the input string represents a node, or nothing in cases where it does not (e.g. if the string is the length of a node).
    */
    match pair.as_rule() {
        Rule::newick => {
            // newick = { subtree ~ ";" }
            // For now, we only do the implementation for one single tree.
            // This case is trivial : we just call handle_pair on the only tree we found.
            for inner in pair.into_inner() {
                if inner.as_rule() == Rule::subtree {
                    return handle_pair(inner);
                }
            }
            None
        },
        Rule::subtree => {
            // subtree = { leaf | internal }
            // Subtree is like the choice between an inner node and a leaf. Either way, we need to pass it to handle_pair.
            handle_pair(pair.into_inner().next().unwrap())
        },
        Rule::leaf => {
            // Choose default values for the name and length of the leaf.
            // The defaults are an empty string and a 0.0 length.
            let mut name = String::new();
            let mut length = 0.0;

            // leaf = { NAME? ~ ":"? ~ LENGTH? }
            // Therefore, we use a match statement to handle the cases NAME and LENGTH because ":" is unimportant.
            for inner_pair in pair.into_inner() {
                match inner_pair.as_rule() {
                    Rule::NAME => {
                        name = inner_pair.as_str().to_string();
                    }
                    Rule::LENGTH => {
                        let val = inner_pair.as_str();
                        if let Err(_) = val.parse::<f64>() {
                            println!("Failed to parse LENGTH: {}", val);
                        }
                        length = val.parse::<f64>().unwrap_or(0.0);
                    },
                    
                    _ => {} // Ignore other rules
                }
            }
            let node = Node {
                name: name,
                left_child: None,
                right_child: None,
                parent: None,
                depth: None,
                length: length,
            };
            Some(node)
        },
        Rule::internal => {
            // internal = { "(" ~ subtree ~ "," ~ subtree ~ ")" ~ NAME? ~ ":"? ~ LENGTH? }
            
            // Initialize default values for the name and length of the internal node.
            // The defaults are an empty string and a 0.0 length.
            let mut name = String::new();
            let mut length = 0.0;
        
            let mut first_subtree = None;
            let mut second_subtree = None;
        
            // Iterate through the inner rules without assuming their order
            for inner_pair in pair.into_inner() {
                match inner_pair.as_rule() {
                    Rule::subtree => {
                        let subtree = handle_pair(inner_pair).unwrap();
                        if first_subtree.is_none() {
                            first_subtree = Some(subtree);
                        } else {
                            second_subtree = Some(subtree);
                        }
                    },
                    Rule::NAME => {
                        name = inner_pair.as_str().to_string();
                    },
                    Rule::LENGTH => {
                        let val = inner_pair.as_str();
                        if let Err(_) = val.parse::<f64>() {
                            println!("Failed to parse LENGTH: {}", val);
                        }
                        length = val.parse::<f64>().unwrap_or(0.0);
                    },
                    
                    _ => {} // Ignore other rules
                }
            }
        
            let node = Node {
                name,
                left_child: first_subtree.map(Box::new),
                right_child: second_subtree.map(Box::new),
                parent: None,
                depth: None,
                length,
            };
            Some(node)
        }
        Rule::NAME | Rule::LENGTH => {
            // We should never directly handle these outside their containing rules.
            None
        },
    }
}

// Convert a Tree in "Node" form to a Newick string.
fn node_to_newick(node: &Node) -> String {
    /* Takes a node and returns the corresponding subtree in Newick format.
        --------------------------------
        INPUT:
            - node: the node to convert to Newick format.
        OUTPUT:
            - the Newick representation of the subtree rooted at node.
        Warning: rounds the lengths to 6 decimal places.
    */
    if let (Some(left_child), Some(right_child)) = (&node.left_child, &node.right_child) {
        // This is an internal node with both left and right children.
        format!(
            "({},{}){}:{:.6}",
            node_to_newick(left_child),
            node_to_newick(right_child),
            node.name,
            node.length
        )
    } else {
        // This is a leaf node.
        format!("{}:{:.6}", node.name, node.length)
    }
}
// Convert from FlatNode to Node
fn flat_to_node(flat_tree: &[FlatNode], index: usize, parent_index: Option<usize>) -> Option<Node> {
    /*
    This function converts a flat tree into an arborescent tree recursively.
    To use it, give the flat_tree, as well as the index of the root. The vector will be traversed recursively, 
    following the descendants of each node being examined.
    ----------------------------------
    Input: 
    - A flat tree, the index of a node in the flat tree, and the index of its parent.
    ----------------------------------
    Output:
    - The corresponding node in the arborescent tree.
    ----------------------------------
    Warning: can bug if applied directly to a non-root node, which will be mistaken for a root node.
    */
    let flat_node = &flat_tree[index];
    let left_child = flat_node.left_child.and_then(|i| {
        flat_to_node(flat_tree, i, Some(index)).map(Box::new)
    });
    let right_child = flat_node.right_child.and_then(|i| {
        flat_to_node(flat_tree, i, Some(index)).map(Box::new)});
    
    Some(Node {
        name: flat_node.name.clone(),
        left_child: left_child,
        right_child: right_child,
        parent: parent_index,
        depth: flat_node.depth,
        length: flat_node.length,
    })
}
// Convert from Node to FlatNode
fn node_to_flat(node: &Node, flat_tree: &mut Vec<FlatNode>, parent: Option<usize>) -> usize {
    /* Transforms the arborescent tree into a "linear tree", which is just a vector of nodes with parent and descendants.
    ----------------------------------
    Input:
    - The root node, which contains the whole arborescent tree
    - The flat_tree vector, which is to be filled by the function
    ----------------------------------
    Output:
    - The index of the node currently being added to flat_tree (usize because the indexing of Rust vectors can only be usize).
    */
    let index = flat_tree.len();
    flat_tree.push(FlatNode {
        name: node.name.clone(),
        left_child: None,  // Will fill this in a moment
        right_child: None, // Will fill this in a moment
        parent: parent,
        depth: node.depth,
        length: node.length,
    });

    if let Some(left) = &node.left_child {
        let left_index = node_to_flat(left, flat_tree, Some(index));
        flat_tree[index].left_child = Some(left_index);
    }

    if let Some(right) = &node.right_child {
        let right_index = node_to_flat(right, flat_tree, Some(index));
        flat_tree[index].right_child = Some(right_index);
    }

    index
}
// Remove any given leaf from the tree



// Give the index of the root of the tree
fn find_root(flat_tree: &Vec<FlatNode>, true_leaf: usize) -> usize {
    /*
    Uses an index of a leaf to find the root of the tree.
     */
    let mut current_node = true_leaf;
    let mut current_parent = flat_tree[current_node].parent;
    while current_parent.is_some() {
        current_node = current_parent.unwrap();
        current_parent = flat_tree[current_node].parent;
    }
    current_node
}

fn name_in_vec(node: &FlatNode, vec: &Vec<(String, String)>) -> bool {
    vec.iter().any(|(name, _)| name == &node.name)
}

fn leaves_translation(ale_tree: &[FlatNode]) -> HashMap<String, String> {
    let mut list_translations = HashMap::new();
    for node in ale_tree {
        // Test if the node is a leaf
        if node.left_child.is_none() && node.right_child.is_none() {
            list_translations.insert(node.name.clone(), node.name.clone());
        }
    }
    list_translations
}

fn find_node_from_name(tree: &[FlatNode], name: &str) -> Option<usize> {
    for (index, node) in tree.iter().enumerate() {
        if name == node.name {
            return Some(index);
        }
    }
    None
}

fn save_hashmap_to_csv(hashmap: &HashMap<String, String>, path: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut writer = Writer::from_writer(file);

    // Optionally write headers
    writer.write_record(&["ale", "original"])?;

    for (key, value) in hashmap {
        writer.write_record(&[key, value])?;
    }

    writer.flush()?;
    Ok(())
}

fn give_name(ale_tree: &[FlatNode], original_tree: &[FlatNode], node_index: usize, list_translations: &mut HashMap<String, String>) -> String {
    let node = &ale_tree[node_index];

    // Has the ale node already been translated?
    if let Some(name) = list_translations.get(&node.name) {
        return name.clone();
    } else {
        // The translation has not been made by leaves_translation, so the node is not a leaf.
        // It must have children, and we can find their translations recursively, and
        // obtain the translation from their names
        let left_child_ale_index = node.left_child.expect("Left child is None");
        let right_child_ale_index = node.right_child.expect("Right child is None");

        let left_child_original_name = give_name(ale_tree, original_tree, left_child_ale_index, list_translations);
        let _right_child_original_name = give_name(ale_tree, original_tree, right_child_ale_index, list_translations);
        // Find the left child
        let left_child_original_index = find_node_from_name(original_tree, &left_child_original_name)
            .expect("Did not find the left child in the original tree");
        let left_child_original = &original_tree[left_child_original_index];
        if let Some(parent_index) = left_child_original.parent {
            let original_name = &original_tree[parent_index].name;
            list_translations.insert(node.name.clone(), original_name.clone());
            return original_name.clone();
        }
        else {
            // Show the node index and name
            println!("Node index: {}, Node name: {}", node_index, node.name);
            panic!();
        }
    }
}
fn main() -> Result<(), Box<dyn Error>>{
    // Read the arguments
    let args: Vec<String> = env::args().collect();

    // Ensure the correct number of arguments are provided
    if args.len() != 4 {
        eprintln!("Usage: {} <original_species_tree_path> <ale_species_tree_path> <output_dir>", args[0]);
        eprintln!("Received arguments: {:?}", args);
        panic!("Incorrect number of arguments");
    }

    let original_sp_tree_path = &args[1];
    let ale_sp_tree_path = &args[2];
    let output_dir = &args[3];
    let original_sp_tree = fs::read_to_string(original_sp_tree_path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    let ale_sp_tree = fs::read_to_string(ale_sp_tree_path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    let mut original_sp_tree = NewickParser::parse(Rule::newick, &original_sp_tree)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    let mut ale_sp_tree = NewickParser::parse(Rule::newick, &ale_sp_tree)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    let mut original_sp_tree = newick_to_tree(original_sp_tree.next().unwrap());
    let mut ale_sp_tree = newick_to_tree(ale_sp_tree.next().unwrap());

    let mut flat_original_sp_tree: Vec<FlatNode> = Vec::new();
    let mut flat_ale_sp_tree: Vec<FlatNode> = Vec::new();

    let _ = node_to_flat(&mut original_sp_tree[0], &mut flat_original_sp_tree, None);
    let _ = node_to_flat(&mut ale_sp_tree[0], &mut flat_ale_sp_tree, None);
    // We now have both trees available in the right format for leaves_translation
    let mut list_translations = leaves_translation(&flat_ale_sp_tree);
    // Find the root of the ale tree
    let root_index = find_root(&flat_ale_sp_tree, 0);
    give_name(&flat_ale_sp_tree, &flat_original_sp_tree, root_index, &mut list_translations);
    let output_path = format!("{}/translations.csv", output_dir);
    let _ = save_hashmap_to_csv(&list_translations, &output_path);
    Ok(())
    }