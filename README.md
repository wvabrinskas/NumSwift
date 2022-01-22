# NumSwift

Adds array arithmetic to Swift. 

## Usage 
### Addition 
```
    let test = 10.0
    let testArray = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    let expected = [11.0, 12.0, 13.0, 14.0, 15.0]
    let result = testArray + test
    
    print(result)
```
- You can add a single value to every member of an array
- This will return the `expected` value. 
```
    let testArray = [1.0, 2.0, 3.0, 4.0, 5.0]
    let testArray2 = [10.0, 20.0, 30.0, 40.0, 50.0]

    let expected = [11.0, 22.0, 33.0, 44.0, 55.0]
    let result = testArray + testArray2
    
    print(result)
```
- You can also add a whole array to another array given they are the same length
- This will return the `expected` value. 

### Multiplication 
```
    let test = 10.0
    let testArray = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    let expected = [10.0, 20.0, 30.0, 40.0, 50.0]
    let result = testArray * test
    
    print(result)
```
- You can multiply a single value to every member of an array
- This will return the `expected` value. 
```
    let testArray = [1.0, 2.0, 3.0, 4.0, 5.0]
    let testArray2 = [10.0, 20.0, 30.0, 40.0, 50.0]

    let expected = [10.0, 40.0, 90.0, 160.0, 250.0]
    
    let result = testArray * testArray2
    
    print(result)
```
- You can also multiply a whole array to another array given they are the same length
- This will return the `expected` value. 

### Division 
```
    let test = 10.0
    let testArray = [10.0, 20.0, 30.0, 40.0, 50.0]

    let expected = [1.0, 2.0, 3.0, 4.0, 5.0]
    let result = testArray / test
    
    print(result)
```
- You can divide a single value to every member of an array
- This will return the `expected` value. 
```
    let testArray = [10.0, 40.0, 90.0, 160.0, 250.0]
    let testArray2 = [1.0, 2.0, 3.0, 4.0, 5.0]

    let expected = [10.0, 20.0, 30.0, 40.0, 50.0]

    let result = testArray / testArray2
    
    print(result)
```
- You can also multiply a whole array to another array given they are the same length
- This will return the `expected` value. 

### Scaling 
You can scale an array in a given range to a desired range. 
```
    let testArray = [0.0, 5.0, 10.0, 20.0]
    let expected = [-1.0, -0.5, 0.0, 1.0]
    
    let scaled = testArray.scale(from: 0...20, to: -1...1)
```
