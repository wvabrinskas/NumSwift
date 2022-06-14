//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/19/22.
//

import Foundation

//Courtesy of: https://github.com/palle-k/DL4S-WGAN-GP/blob/master/Sources/WGANGP/Util.swift
extension DispatchQueue {
  static func concurrentPerform(units: Int, workers: Int, priority: DispatchQoS.QoSClass = .default, task: @escaping (Int) -> ()) {
    let sema = DispatchSemaphore(value: 0)
    let tasksPerWorker = units / workers
        
    for workerID in 0 ..< workers {
      let taskRange: CountableRange<Int>
      if workerID == workers - 1 {
        taskRange = tasksPerWorker * workerID ..< units
      } else {
        taskRange = tasksPerWorker * workerID ..< tasksPerWorker * (workerID + 1)
      }
      
      DispatchQueue.global(qos: priority).async {
        for unitID in taskRange {
          task(unitID)
        }
        
        sema.signal()
      }
    }
    
    for _ in 0 ..< workers {
      sema.wait()
    }
  }
  
  static func concurrentPerform<Result>(units: Int, workers: Int, priority: DispatchQoS.QoSClass = .default, task: @escaping (Int) -> Result) -> [Result] {
    let sema = DispatchSemaphore(value: 0)
    let tasksPerWorker = units / workers
    
    var results = [Result?](repeating: nil, count: units)
    
    for workerID in 0 ..< workers {
      let taskRange: CountableRange<Int>
      if workerID == workers - 1 {
        taskRange = tasksPerWorker * workerID ..< units
      } else {
        taskRange = tasksPerWorker * workerID ..< tasksPerWorker * (workerID + 1)
      }
      
      DispatchQueue.global(qos: priority).async {
        for unitID in taskRange {
          results[unitID] = task(unitID)
        }
        
        sema.signal()
      }
    }
    
    for _ in 0 ..< workers {
      sema.wait()
    }
    
    return results.compactMap {$0}
  }
}
